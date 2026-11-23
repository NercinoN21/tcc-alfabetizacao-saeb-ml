"""Valor incremental dos blocos (ablação), robustez (Seção 3.9), sensibilidades, proxies e equidade.

Pergunta central do TCC 2 (alinhada à H3): quanto os questionários de professor e
diretor acrescentam, FORA DA AMOSTRA, além do INSE e da geografia? Cada especificação é comparada
ao modelo completo nas MESMAS dobras, com IC da diferença pelo t corrigido de Nadeau–Bengio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold

from .avaliacao import cv_repetida, ic_diferenca_pareada
from .modelos import X_de, montar_preprocessador, selecionar_variaveis


def blocos(meta_sel: pd.DataFrame) -> dict[str, list[str]]:
    v = meta_sel.variavel.tolist()
    return {"inse": [x for x in v if x == "ESC_INSE"],
            "escola": [x for x in v if x == "ESC_PC_FORMACAO_DOCENTE_INICIAL"],
            "geo": [x for x in v if x.startswith("GEO_")],
            "prof": [x for x in v if x.startswith("PROF_") or x == "CTX_N_TURMAS"],
            "dir": [x for x in v if x.startswith("DIR_")]}


def ablacao(pop, meta_sel, alvo, trilha, nome, params, cfg) -> pd.DataFrame:
    b = blocos(meta_sel)
    tudo = meta_sel.variavel.tolist()
    specs = {
        "M0: só INSE": b["inse"],
        "M1: INSE + geografia": b["inse"] + b["geo"],
        "M2: M1 + professor": b["inse"] + b["geo"] + b["prof"],
        "M3: M1 + diretor": b["inse"] + b["geo"] + b["dir"],
        "M4: completo": tudo,
        "M5: completo sem INSE": [x for x in tudo if x not in b["inse"] + b["escola"]],
        "M6: só professor + diretor": b["prof"] + b["dir"],
        "M4 − professor": [x for x in tudo if x not in b["prof"]],
        "M4 − diretor": [x for x in tudo if x not in b["dir"]],
        "M4 − geografia": [x for x in tudo if x not in b["geo"]],
        "M7: M1 + índices por tema": tudo,          # especificação extra declarada na D1 (antes da execução final)
    }
    res = {k: cv_repetida(pop, meta_sel[meta_sel.variavel.isin(cols)], alvo, trilha, nome, params, cfg,
                          indices_tema=k.startswith("M7"))
           for k, cols in specs.items()}
    ref, ref1 = res["M4: completo"]["mae_dobras"], res["M1: INSE + geografia"]["mae_dobras"]
    linhas = {}
    for k, r in res.items():
        ic = ic_diferenca_pareada(r["mae_dobras"] - ref)            # > 0  => pior que o completo
        ic1 = ic_diferenca_pareada(r["mae_dobras"] - ref1)          # < 0  => melhor que INSE + geografia
        linhas[k] = {"n variáveis": len(specs[k]), "MAE (CV)": r["MAE"], "MAE baseline": r["MAE_baseline"],
                     "ganho vs baseline (%)": r["ganho_%"], "R² (CV)": r["R2"],
                     "Δ MAE vs M4": ic["media"], "IC95% inf": ic["ic_inf"], "IC95% sup": ic["ic_sup"],
                     "Δ MAE vs M1": ic1["media"], "IC95% inf vs M1": ic1["ic_inf"], "IC95% sup vs M1": ic1["ic_sup"]}
    return pd.DataFrame(linhas).T.infer_objects()


def tres_execucoes(pop, universo, meta, alvo, trilha, nome, params, cfg) -> pd.DataFrame:
    """R1 pública com INSE · R2 pública sem INSE · R3 universo sem INSE (varia um fator por vez)."""
    m1, _ = selecionar_variaveis(pop, meta, com_cor=False, limite_ausencia=cfg.limite_ausencia,
                                 excluir_percepcao=cfg.excluir_percepcao_desfecho)
    m2, _ = selecionar_variaveis(pop, meta, com_cor=False, com_inse=False, limite_ausencia=cfg.limite_ausencia,
                                 excluir_percepcao=cfg.excluir_percepcao_desfecho)
    m3, _ = selecionar_variaveis(universo, meta, com_cor=False, com_inse=False, limite_ausencia=cfg.limite_ausencia,
                                 excluir_percepcao=cfg.excluir_percepcao_desfecho)
    linhas = []
    for rot, df, ms in (("R1: pública com INSE", pop, m1), ("R2: pública sem INSE", pop, m2),
                        ("R3: universo sem INSE", universo, m3)):
        r = cv_repetida(df, ms, alvo, trilha, nome, params, cfg)
        linhas.append({"execução": rot, "escolas": len(df), "variáveis": len(ms), "MAE": r["MAE"],
                       "MAE baseline": r["MAE_baseline"], "ganho sobre baseline (%)": r["ganho_%"], "R²": r["R2"]})
    return pd.DataFrame(linhas).set_index("execução")


def construir_base_turma(alunos_validos, prof_cod_turma, pop, n_min_turma: int = 5) -> pd.DataFrame:
    """Turma como unidade: desfechos por turma; professor da própria turma; resto herdado da escola."""
    a = alunos_validos[alunos_validos.ID_ESCOLA.isin(pop.index)]
    t = a.groupby("ID_TURMA").agg(N_ALUNOS=("ALFAB", "size"), Y_PROP_ALFAB=("ALFAB", "mean"),
                                  Y_PROF_MEDIA=("PROFICIENCIA_LP_SAEB", "mean"), ID_ESCOLA=("ID_ESCOLA", "first"))
    t = t[t.N_ALUNOS >= n_min_turma]
    pc = prof_cod_turma.drop_duplicates("ID_TURMA").set_index("ID_TURMA").drop(columns=["ID_ESCOLA"])
    t = t.join(pc, how="inner")
    esc = [c for c in pop.columns if c.startswith(("GEO_", "DIR_", "ESC_")) or c in ("CTX_N_TURMAS", "ID_MUNICIPIO")]
    return t.join(pop[esc], on="ID_ESCOLA")


def sensibilidades(pop, base_todas, base_interna, base_turma, meta, alvo, trilha, nome, params, cfg) -> pd.DataFrame:
    sel = lambda df: selecionar_variaveis(df, meta, com_cor=False, limite_ausencia=cfg.limite_ausencia,
                                          excluir_percepcao=cfg.excluir_percepcao_desfecho)[0]
    ms = sel(pop)
    w = pop["PESO"].values
    linhas = []

    def add(eixo, variante, df, m, **kw):
        r = cv_repetida(df, m, alvo, trilha, nome, params, cfg, **kw)
        d = {"eixo": eixo, "variante": variante, "linhas": len(df), "variáveis": len(m), "MAE": r["MAE"],
             "MAE baseline": r["MAE_baseline"], "ganho sobre baseline (%)": r["ganho_%"], "R²": r["R2"]}
        if "MAE_pond" in r:
            d.update({"MAE ponderado": r["MAE_pond"], "MAE baseline ponderado": r["MAE_baseline_pond"],
                      "ganho ponderado (%)": 100 * (1 - r["MAE_pond"] / r["MAE_baseline_pond"])})
        linhas.append(d)

    add("CV", "5-dobras por escola (referência)", pop, ms)
    add("CV", "GroupKFold por município", pop, ms, grupos=pop.ID_MUNICIPIO)
    add("Unidade", "turma (CV agrupada por escola)", base_turma, sel(base_turma), grupos=base_turma.ID_ESCOLA)
    for n in (1, 5, 10, 15, 20):
        p = base_todas[(base_todas.PUBLICA == 1) & base_todas.TEM_INSE & (base_todas.N_ALUNOS >= n)]
        add("n mínimo", f"n ≥ {n}", p, sel(p))
    pi = base_interna[base_interna.N_ALUNOS >= cfg.n_min_alunos]
    add("Junção", "interna (professor E diretor)", pi, sel(pi))
    com_percepcao = selecionar_variaveis(pop, meta, com_cor=False, limite_ausencia=cfg.limite_ausencia,
                                         excluir_percepcao=False)[0]
    add("Percepção do desfecho", "COM os itens PROF_TX_Q080 e DIR_TX_Q030", pop, com_percepcao)
    add("Pesos", "treino sem peso amostral · avaliação ponderada", pop, ms, w_aval=w)
    add("Pesos", "treino COM peso amostral · avaliação ponderada", pop, ms, w=w, w_aval=w)
    add("Pesos", "treino com peso truncado no p99 · avaliação ponderada", pop, ms,
        w=np.minimum(w, np.quantile(w, .99)), w_aval=w)
    return pd.DataFrame(linhas).set_index(["eixo", "variante"])


def deff_kish(w) -> float:
    """Efeito de desenho aproximado pela variação dos pesos: deff = n·Σw² / (Σw)² (1 = pesos iguais)."""
    w = np.asarray(w, float)
    return float(len(w) * np.sum(w ** 2) / np.sum(w) ** 2)


def diagnostico_pesos(pop, variaveis=("Y_PROP_ALFAB", "Y_PROF_MEDIA", "ESC_INSE")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnóstico do peso amostral da escola (Seção 3.3): dispersão, deff de Kish, escolas efetivas e
    contraste entre média ponderada e não ponderada (Solon; Haider; Wooldridge, 2015)."""
    from scipy.stats import spearmanr
    w = pop["PESO"].values
    deff = deff_kish(w)
    resumo = pd.Series({
        "escolas": len(w), "pesos imputados (sem questionário do diretor, %)": 100 * float(pop["PESO_ESCOLA"].isna().mean()),
        "coeficiente de variação do peso": float(np.std(w) / np.mean(w)), "deff de Kish": deff,
        "escolas efetivas (n / deff)": len(w) / deff, "razão peso máximo / mínimo": float(w.max() / w.min()),
        "razão p99 / p1": float(np.quantile(w, .99) / np.quantile(w, .01))}, name="valor").to_frame()
    linhas = []
    for v in variaveis:
        x = pop[v].values.astype(float)
        ok = ~np.isnan(x)
        m, mp = float(x[ok].mean()), float(np.average(x[ok], weights=w[ok]))
        linhas.append({"variável": v, "média sem peso": m, "média ponderada": mp,
                       "diferença relativa (%)": 100 * (mp / m - 1),
                       "Spearman peso × variável": float(spearmanr(w[ok], x[ok]).statistic)})
    return resumo, pd.DataFrame(linhas).set_index("variável")


def alfabetizacao_por_rede(aluno: pd.DataFrame, corte: float = 743) -> pd.DataFrame:
    """Proporção de alunos do 2º ano no corte de alfabetização ou acima, por rede, com e sem o peso amostral
    do aluno (PESO_ALUNO_LP). Universo: alunos com proficiência em LP calculada (IN_PROFICIENCIA_LP = 1).
    Número de contexto da Introdução, calculado nos microdados (fonte primária)."""
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")
    a = aluno.assign(_w=num(aluno["PESO_ALUNO_LP"]), _y=num(aluno["PROFICIENCIA_LP_SAEB"]))
    a = a[(a["IN_PROFICIENCIA_LP"] == 1) & a["_w"].notna() & a["_y"].notna()]
    alf = (a["_y"] >= corte).astype(float)
    linhas = []
    for rot, m in (("total", pd.Series(True, index=a.index)), ("pública", a["IN_PUBLICA"] == 1),
                   ("privada", a["IN_PUBLICA"] == 0)):
        linhas.append({"rede": rot, "alunos": int(m.sum()),
                       "proporção com peso (%)": 100 * float(np.average(alf[m], weights=a.loc[m, "_w"])),
                       "proporção sem peso (%)": 100 * float(alf[m].mean())})
    return pd.DataFrame(linhas).set_index("rede")


def teste_proxies(pop, meta_sel, cfg) -> pd.DataFrame:
    """Dá para recuperar cor/raça a partir das variáveis do cenário A? (excluir ≠ remover a informação)"""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.pipeline import Pipeline

    kw = dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10, colsample_bytree=0.5,
              subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1, random_state=cfg.seed)
    d = pop[pop.PROF_COR_PCT_PRETA_PARDA.notna()]
    X, y = X_de(d, meta_sel), d.PROF_COR_PCT_PRETA_PARDA.values
    pred = np.zeros(len(d))
    for a_, b_ in KFold(5, shuffle=True, random_state=cfg.seed).split(X):
        pred[b_] = Pipeline([("pre", montar_preprocessador(meta_sel, False)), ("model", LGBMRegressor(**kw))]
                            ).fit(X.iloc[a_], y[a_]).predict(X.iloc[b_])
    d2 = pop[pop.DIR_COR.isin(["Branca", "Preta", "Parda"])]
    X2, y2 = X_de(d2, meta_sel), d2.DIR_COR.isin(["Preta", "Parda"]).astype(int).values
    prob = np.zeros(len(d2))
    for a_, b_ in KFold(5, shuffle=True, random_state=cfg.seed).split(X2):
        prob[b_] = Pipeline([("pre", montar_preprocessador(meta_sel, False)), ("model", LGBMClassifier(**kw))]
                            ).fit(X2.iloc[a_], y2[a_]).predict_proba(X2.iloc[b_])[:, 1]
    return pd.DataFrame([
        {"alvo do teste": "% turmas com professor preto/pardo", "n": len(d), "R² (CV)": r2_score(y, pred),
         "MAE modelo": mean_absolute_error(y, pred), "MAE chute (média)": mean_absolute_error(y, np.full(len(y), y.mean())),
         "AUC": np.nan},
        {"alvo do teste": "diretor preto/pardo (vs branco)", "n": len(d2), "R² (CV)": np.nan, "MAE modelo": np.nan,
         "MAE chute (média)": np.nan, "AUC": roc_auc_score(y2, prob)},
    ]).set_index("alvo do teste")


def grupo_inse(v) -> str:
    return "baixo (I–III)" if v <= 3 else ("médio (IV)" if v == 4 else "alto (V–VIII)")


def auditar_equidade(pop, alvo, pred_a, pred_b, cfg):
    """Erro por grupo (cenários A e B) e gap de MAE entre grupos, com IC bootstrap da mudança do gap (B − A).
    Os ICs ignoram a incerteza de re-treino: tratá-los como indicativos."""
    y = pop[alvo].values
    ea, eb = pred_a - y, pred_b - y
    inse = pop.ESC_INSE.map(grupo_inse)
    dirc = pop.DIR_COR.map({"Branca": "branco", "Preta": "preto/pardo", "Parda": "preto/pardo"})
    prof = pd.Series(np.where(pop.PROF_COR_PCT_PRETA_PARDA.isna(), None,
                              np.where(pop.PROF_COR_PCT_PRETA_PARDA > 0.5, "maioria preta/parda", "minoria preta/parda")),
                     index=pop.index)
    eixos = [("INSE", inse, ["baixo (I–III)", "médio (IV)", "alto (V–VIII)"], "baixo (I–III)", "alto (V–VIII)"),
             ("Cor do diretor", dirc, ["preto/pardo", "branco"], "preto/pardo", "branco"),
             ("Cor do(s) professor(es)", prof, ["maioria preta/parda", "minoria preta/parda"],
              "maioria preta/parda", "minoria preta/parda")]
    rng = np.random.default_rng(cfg.seed)
    grupos, gaps = [], []
    for nome, g, niveis, g1, g0 in eixos:
        for lab in niveis:
            m = (g == lab).values
            grupos.append({"eixo": nome, "grupo": lab, "n": int(m.sum()), "MAE A": np.abs(ea[m]).mean(),
                           "MAE B": np.abs(eb[m]).mean(), "viés A (prev−obs)": ea[m].mean(), "viés B (prev−obs)": eb[m].mean()})
        i1, i0 = np.where((g == g1).values)[0], np.where((g == g0).values)[0]
        ga = np.abs(ea[i1]).mean() - np.abs(ea[i0]).mean()
        gb = np.abs(eb[i1]).mean() - np.abs(eb[i0]).mean()
        boots = []
        for _ in range(cfg.n_boot):
            b1, b0 = rng.choice(i1, len(i1)), rng.choice(i0, len(i0))
            boots.append((np.abs(eb[b1]).mean() - np.abs(eb[b0]).mean()) - (np.abs(ea[b1]).mean() - np.abs(ea[b0]).mean()))
        gaps.append({"eixo": nome, "gap": f"MAE({g1}) − MAE({g0})", "gap A": ga, "gap B": gb, "Δ gap (B − A)": gb - ga,
                     "IC95% inf": np.percentile(boots, 2.5), "IC95% sup": np.percentile(boots, 97.5)})
    return pd.DataFrame(grupos).set_index(["eixo", "grupo"]), pd.DataFrame(gaps).set_index("eixo")
