"""Explicabilidade do modelo selecionado (Seção 3.8): SHAP global/temático, estabilidade e LIME local.

SHAP: TreeExplainer para ensembles; LinearExplainer se o modelo selecionado for o linear.
Escala: margem do modelo (log-odds na trilha de proporção) -> reporta-se a PARTICIPAÇÃO (%) no |SHAP|.
Colunas one-hot são reagrupadas na variável de origem. Por tema, reporta-se a soma E a média por
variável (a soma infla temas com muitos itens).

LIME (checagem local complementar, elogiada pelo avaliador): para casos escolhidos por regra
PRÉ-DECLARADA (não por conveniência), compara-se a explicação LIME com a decomposição SHAP do mesmo
caso, medindo (i) fidelidade local do LIME (R² do modelo substituto), (ii) estabilidade do LIME entre
sementes (Jaccard médio do top-k) e (iii) concordância LIME × SHAP (Jaccard do top-k, Spearman das
atribuições e concordância de sinal). Divergências são resultado a discutir (Doshi-Velez; Kim, 2017).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .modelos import X_de, pipeline_de, particao, variavel_original


# ------------------------------------------------------------------ SHAP
def _explicador(mdl, Z_fundo):
    import shap

    if hasattr(mdl, "coef_") and not hasattr(mdl, "estimators_") and not hasattr(mdl, "booster_") \
            and not hasattr(mdl, "get_booster"):
        coef = np.ravel(mdl.coef_)
        inter = float(np.ravel(getattr(mdl, "intercept_", 0.0))[0])
        fundo = shap.maskers.Independent(Z_fundo, max_samples=len(Z_fundo))   # todo o treino, determinístico
        return shap.LinearExplainer((coef, inter), fundo)
    return shap.TreeExplainer(mdl)


def shap_local(pipe, base, meta_sel, idx, idx_fundo) -> pd.DataFrame:
    """Valores SHAP COM SINAL por escola × variável original (one-hot somado). Linhas = escolas `idx`."""
    pre, mdl = pipe.named_steps["pre"], pipe.named_steps["model"]
    Z = pre.transform(X_de(base.loc[idx], meta_sel))
    Zf = pre.transform(X_de(base.loc[idx_fundo], meta_sel))
    sv = np.asarray(_explicador(mdl, Zf).shap_values(Z))
    orig = variavel_original(pre.get_feature_names_out(), meta_sel)
    return pd.DataFrame(sv, columns=orig, index=idx).T.groupby(level=0).sum().T


def shap_por_variavel(pipe, base, meta_sel, idx, idx_fundo) -> pd.Series:
    """Participação (%) de cada variável original no |SHAP| total, nas escolas `idx`."""
    pre, mdl = pipe.named_steps["pre"], pipe.named_steps["model"]
    Z = pre.transform(X_de(base.loc[idx], meta_sel))
    Zf = pre.transform(X_de(base.loc[idx_fundo], meta_sel))
    sv = np.abs(np.asarray(_explicador(mdl, Zf).shap_values(Z)))
    orig = variavel_original(pre.get_feature_names_out(), meta_sel)
    imp = pd.DataFrame(sv, columns=orig).T.groupby(level=0).sum().T.mean()
    return (imp / imp.sum()).sort_values(ascending=False)


def shap_por_tema(imp_var: pd.Series, meta_sel: pd.DataFrame) -> pd.DataFrame:
    tema = meta_sel.set_index("variavel")["tema"]
    t = imp_var.groupby(imp_var.index.map(tema)).agg(["sum", "count"])
    t.columns = ["participacao", "n_variaveis"]
    t["participacao_por_variavel"] = t["participacao"] / t["n_variaveis"]
    return t.sort_values("participacao", ascending=False)


def estabilidade(pop, meta_sel, alvo, trilha, nome, params, cfg) -> pd.DataFrame:
    """Repete partição → ajuste (hiperparâmetros fixos) → SHAP no teste, com `cfg.repet_estab` sementes."""
    cols = {}
    for s in range(cfg.repet_estab):
        tr, te = particao(pop, "Y_PROP_ALFAB", cfg.seed + 100 + s, cfg.teste)
        pipe = pipeline_de(trilha, nome, meta_sel, cfg.seed + s, params, cfg.n_arvores_rf)
        kw = {"model__sample_weight": pop.loc[tr, "N_ALUNOS"].values} if trilha == "prop" else {}
        pipe.fit(X_de(pop.loc[tr], meta_sel), pop.loc[tr, alvo].values, **kw)
        cols[s] = shap_por_tema(shap_por_variavel(pipe, pop, meta_sel, te, tr), meta_sel)["participacao"]
    m = pd.DataFrame(cols).fillna(0)
    r = m.rank(ascending=False)
    return pd.DataFrame({"participação média": m.mean(axis=1), "posição média": r.mean(axis=1),
                         "dp da posição": r.std(axis=1), "% repetições no top-3": 100 * (r <= 3).mean(axis=1)}
                        ).sort_values("posição média")


# ------------------------------------------------------------------ LIME
CASOS = ("típica (menor |resíduo|)", "acima do esperado (maior resíduo +)", "abaixo do esperado (maior resíduo −)",
         "limítrofe (desfecho ≈ mediana)", "extremo inferior (menor desfecho)", "extremo superior (maior desfecho)")


def selecionar_casos(y: pd.Series, p: pd.Series) -> dict[str, object]:
    """Regra pré-declarada de seleção de escolas do TESTE para explicação local (sem repetição)."""
    res = y - p
    escolhidos, usados = {}, set()

    def pega(ordem, rotulo):
        for i in ordem:
            if i not in usados:
                escolhidos[rotulo] = i
                usados.add(i)
                return

    pega(res.abs().sort_values().index, CASOS[0])
    pega(res.sort_values(ascending=False).index, CASOS[1])
    pega(res.sort_values().index, CASOS[2])
    pega((y - y.median()).abs().sort_values().index, CASOS[3])
    pega(y.sort_values().index, CASOS[4])
    pega(y.sort_values(ascending=False).index, CASOS[5])
    return escolhidos


def _lime_por_variavel(expl, z, prever, n_feat, orig, num_samples) -> tuple[pd.Series, float]:
    """Pesos LIME reagrupados na variável original + R² local do modelo substituto."""
    e = expl.explain_instance(z, prever, num_features=n_feat, num_samples=num_samples)
    pares = e.as_map()[1]          # regressão: a chave 1 tem o sinal correto (a chave 0 é o espelho negativo)
    pesos = pd.Series([w for _, w in pares], index=[orig[i] for i, _ in pares])
    return pesos.groupby(level=0).sum(), float(e.score)


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def lime_vs_shap(pipe, base, meta_sel, idx_tr, idx_te, alvo, cfg, k: int = 10, n_sementes: int = 10,
                 num_samples: int = 5000) -> tuple[pd.DataFrame, dict]:
    """Para cada caso pré-declarado: fidelidade, estabilidade do LIME e concordância com o SHAP."""
    from lime.lime_tabular import LimeTabularExplainer

    pre, mdl = pipe.named_steps["pre"], pipe.named_steps["model"]
    Ztr = pre.transform(X_de(base.loc[idx_tr], meta_sel))
    Zte = pd.DataFrame(pre.transform(X_de(base.loc[idx_te], meta_sel)), index=idx_te)
    nomes = list(pre.get_feature_names_out())
    orig = variavel_original(nomes, meta_sel)
    categoricas = [i for i, n in enumerate(nomes) if n.startswith("nom__")]
    prever = lambda Z: np.asarray(mdl.predict(Z), float)
    y = base.loc[idx_te, alvo]
    p = pd.Series(prever(Zte.values), index=idx_te)
    casos = selecionar_casos(y, p)
    sv = shap_local(pipe, base, meta_sel, list(casos.values()), idx_tr)
    linhas, detalhes = [], {}
    for rotulo, i in casos.items():
        execs, scores = [], []
        for s in range(n_sementes):
            expl = LimeTabularExplainer(Ztr, mode="regression", feature_names=nomes, categorical_features=categoricas,
                                        discretize_continuous=True, random_state=cfg.seed + s)
            w, sc = _lime_por_variavel(expl, Zte.loc[i].values, prever, 3 * k, orig, num_samples)
            execs.append(w)
            scores.append(sc)
        tops = [w.abs().sort_values(ascending=False).head(k).index for w in execs]
        estab = float(np.mean([_jaccard(a, b) for a, b in itertools.combinations(tops, 2)]))
        lime_med = pd.concat(execs, axis=1).fillna(0).mean(axis=1)
        top_lime = lime_med.abs().sort_values(ascending=False).head(k).index
        s_i = sv.loc[i]
        top_shap = s_i.abs().sort_values(ascending=False).head(k).index
        uniao = top_lime.union(top_shap)
        inter = top_lime.intersection(top_shap)
        rho = stats.spearmanr(lime_med.reindex(uniao).fillna(0), s_i.reindex(uniao).fillna(0))[0]
        sinal = float(np.mean(np.sign(lime_med[inter]) == np.sign(s_i[inter]))) if len(inter) else np.nan
        linhas.append({"caso": rotulo, "escola": i, "observado": float(y[i]), "previsto": float(p[i]),
                       "fidelidade_LIME_R2_media": float(np.mean(scores)), "estabilidade_LIME_jaccard_topk": estab,
                       "concordancia_jaccard_topk": _jaccard(top_lime, top_shap), "spearman_LIME_SHAP": float(rho),
                       "concordancia_de_sinal": sinal, "top3_SHAP": ", ".join(top_shap[:3]),
                       "top3_LIME": ", ".join(top_lime[:3])})
        detalhes[rotulo] = pd.DataFrame({"SHAP": s_i.reindex(uniao), "LIME_media": lime_med.reindex(uniao)}).fillna(0)
    return pd.DataFrame(linhas).set_index("caso"), detalhes
