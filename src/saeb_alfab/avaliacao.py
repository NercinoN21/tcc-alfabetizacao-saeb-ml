"""Métricas, competição de modelos, CV repetida, regra de seleção do modelo final e pinball.

REGRA DE SELEÇÃO DO MODELO FINAL (Seção 3.7, agora implementada como escrita; decidida só no TREINO):
  d_k = MAE_linear,k − MAE_ensemble,k  nas mesmas dobras k de uma CV 5-dobras repetida;
  IC95% de média(d) pelo t corrigido de Nadeau e Bengio (2003), que corrige a dependência entre
  dobras sobrepostas: Var = s²_d · (1/J + n_val/n_trein).
  O ENSEMBLE é selecionado sse  IC_inf > 0  E  média(d)/MAE_linear ≥ margem_equivalencia.
  Caso contrário, seleciona-se o LINEAR (mais simples e intrinsecamente interpretável).
O conjunto de teste é tocado uma única vez, para reportar o modelo selecionado (com IC bootstrap).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV

from .config import QUANTIS, Config
from .modelos import ENSEMBLES, LINEAR, MODELOS, X_de, especificar, montar_preprocessador, pipeline_de, usa_peso_n


# ------------------------------------------------------------------ métricas
def smape(y, p) -> float:
    """sMAPE (%) = 100 · média( 2|y − p| / (|y| + |p|) ); pares com denominador zero são ignorados."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    ok = den > 0
    return float(100 * np.mean(2 * np.abs(y[ok] - p[ok]) / den[ok]))


def metricas(y, p, n=None, trilha: str = "prof") -> dict:
    d = dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))
    if trilha == "prof":                 # sMAPE só na proficiência (na proporção, y = 0 o torna instável)
        d["sMAPE_%"] = smape(y, p)
    if n is not None:
        d["MAE_pond_n"] = mean_absolute_error(y, p, sample_weight=n)
    return d


def ic_bootstrap_mae(y, p, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    e = np.abs(np.asarray(y) - np.asarray(p))
    bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _clip(trilha, p):
    p = np.asarray(p, float)
    return np.clip(p, 0, 1) if trilha == "prop" else p


# ------------------------------------------------------------------ competição (Seção 3.6.4)
def competir(base, meta_sel, alvo, trilha, idx_tr, idx_te, cfg: Config, cv_grupos=None):
    """Baselines + 4 modelos. Hiperparâmetros: busca aleatória só no treino (CV 5-dobras por escola,
    ou GroupKFold se `cv_grupos`), critério MAE. O teste é tocado uma vez, para reportar."""
    y = base[alvo]
    Xtr, Xte = X_de(base.loc[idx_tr], meta_sel), X_de(base.loc[idx_te], meta_sel)
    ytr, yte = y.loc[idx_tr].values, y.loc[idx_te].values
    n_tr, n_te = base.loc[idx_tr, "N_ALUNOS"].values, base.loc[idx_te, "N_ALUNOS"].values
    cortes = (list(KFold(5, shuffle=True, random_state=cfg.seed).split(Xtr)) if cv_grupos is None
              else list(GroupKFold(5).split(Xtr, ytr, groups=cv_grupos.loc[idx_tr].values)))
    linhas, ajustados, preds = [], {}, {}
    for nome, estr in (("Baseline: média", "mean"), ("Baseline: mediana", "median")):
        p = DummyRegressor(strategy=estr).fit(Xtr, ytr).predict(Xte)
        linhas.append(dict(modelo=nome, cv_MAE=np.nan, **metricas(yte, p, n_te, trilha)))
        preds[nome] = p
    for nome, (est, dist, escalar, peso_n) in especificar(trilha, cfg.seed, cfg.n_arvores_rf).items():
        t0 = time.time()
        pipe = pipeline_de(trilha, nome, meta_sel, cfg.seed, n_arvores=cfg.n_arvores_rf)
        busca = RandomizedSearchCV(pipe, dist, n_iter=cfg.n_iter, cv=cortes, scoring="neg_mean_absolute_error",
                                   random_state=cfg.seed, n_jobs=-1, refit=True)
        busca.fit(Xtr, ytr, **({"model__sample_weight": n_tr} if peso_n else {}))
        p_bruto = np.asarray(busca.predict(Xte), float)
        fora = float(np.mean((p_bruto < 0) | (p_bruto > 1))) if trilha == "prop" else np.nan
        p = _clip(trilha, p_bruto)
        ic = ic_bootstrap_mae(yte, p, cfg.n_boot, cfg.seed)
        linhas.append(dict(modelo=nome, cv_MAE=-busca.best_score_, **metricas(yte, p, n_te, trilha),
                           MAE_IC95_inf=ic[0], MAE_IC95_sup=ic[1], pct_fora_de_0_1=fora,
                           segundos=round(time.time() - t0, 1)))
        ajustados[nome], preds[nome] = busca, p
    res = pd.DataFrame(linhas).set_index("modelo")
    res["ganho_vs_baseline_media_%"] = 100 * (1 - res["MAE"] / res.loc["Baseline: média", "MAE"])
    return res, ajustados, preds


def params_busca(busca) -> dict:
    return {k.replace("model__", ""): v for k, v in busca.best_params_.items()}


# ------------------------------------------------------------------ CV repetida (OOF, ablação, sensibilidades, seleção)
def cv_repetida(base, meta_sel, alvo, trilha, nome, params, cfg: Config, grupos=None, w=None, w_aval=None,
                repet: int | None = None) -> dict:
    """CV 5-dobras repetida com hiperparâmetros fixos (GroupKFold: 1 passada). `w` = pesos de treino
    (padrão: peso de precisão n, se o modelo o usa); `w_aval` = pesos só da avaliação.
    Devolve MAE médio, baseline, R², ganho, MAE por dobra (para comparações pareadas) e OOF médio."""
    y, X, n = base[alvo].values, X_de(base, meta_sel), len(base)
    repet = 1 if grupos is not None else (repet or cfg.repet_cv)
    pred, basel = np.full((repet, n), np.nan), np.full((repet, n), np.nan)
    mae_dobras, mae_dobras_base = [], []
    peso_n = usa_peso_n(trilha, nome)
    for r in range(repet):
        cortes = (KFold(5, shuffle=True, random_state=cfg.seed + r).split(X) if grupos is None
                  else GroupKFold(5).split(X, y, groups=np.asarray(grupos)))
        for a_, b_ in cortes:
            pipe = pipeline_de(trilha, nome, meta_sel, cfg.seed + r, params, cfg.n_arvores_rf)
            sw = w[a_] if w is not None else (base["N_ALUNOS"].values[a_] if peso_n else None)
            pipe.fit(X.iloc[a_], y[a_], **({} if sw is None else {"model__sample_weight": sw}))
            pred[r, b_] = _clip(trilha, pipe.predict(X.iloc[b_]))
            basel[r, b_] = y[a_].mean()
            mae_dobras.append(mean_absolute_error(y[b_], pred[r, b_]))
            mae_dobras_base.append(mean_absolute_error(y[b_], basel[r, b_]))
    err, errb = np.abs(pred - y), np.abs(basel - y)
    out = dict(MAE=float(err.mean()), MAE_baseline=float(errb.mean()),
               R2=float(np.mean([r2_score(y, pr) for pr in pred])),
               mae_dobras=np.array(mae_dobras), mae_dobras_baseline=np.array(mae_dobras_base),
               pred=pred.mean(axis=0), n_val_sobre_n_trein=0.25)
    out["ganho_%"] = 100 * (1 - out["MAE"] / out["MAE_baseline"])
    wa = w_aval if w_aval is not None else w
    if wa is not None:
        out["MAE_pond"] = float(np.mean([np.average(e, weights=wa) for e in err]))
        out["MAE_baseline_pond"] = float(np.mean([np.average(e, weights=wa) for e in errb]))
    return out


def ic_diferenca_pareada(d: np.ndarray, razao_val_trein: float = 0.25, alpha: float = 0.05) -> dict:
    """Média e IC da diferença entre dobras pareadas, com a correção de Nadeau e Bengio (2003)."""
    d = np.asarray(d, float)
    J = len(d)
    var = d.var(ddof=1) * (1 / J + razao_val_trein)
    ep = np.sqrt(var)
    t = stats.t.ppf(1 - alpha / 2, J - 1)
    m = float(d.mean())
    p = float(2 * stats.t.sf(abs(m / ep), J - 1)) if ep > 0 else 0.0
    return dict(media=m, ep=float(ep), ic_inf=m - t * ep, ic_sup=m + t * ep, p_corrigido=p, n_dobras=J)


def regra_selecao(mae_linear: np.ndarray, mae_ensemble: np.ndarray, margem: float) -> dict:
    """Aplica a regra pré-registrada. Função pura (testável)."""
    ic = ic_diferenca_pareada(np.asarray(mae_linear) - np.asarray(mae_ensemble))
    ganho_rel = ic["media"] / float(np.mean(mae_linear))
    escolhe_ensemble = (ic["ic_inf"] > 0) and (ganho_rel >= margem)
    return dict(**ic, ganho_relativo=ganho_rel, margem=margem, ensemble_selecionado=bool(escolhe_ensemble))


def selecionar_modelo_final(base, meta_sel, alvo, trilha, idx_tr, ajustados, res, cfg: Config) -> dict:
    """Melhor ensemble por cv_MAE × linear, comparados em CV repetida no TREINO com as mesmas dobras."""
    melhor_ens = res.loc[ENSEMBLES, "cv_MAE"].idxmin()
    tr = base.loc[idx_tr]
    r_lin = cv_repetida(tr, meta_sel, alvo, trilha, LINEAR, params_busca(ajustados[LINEAR]), cfg)
    r_ens = cv_repetida(tr, meta_sel, alvo, trilha, melhor_ens, params_busca(ajustados[melhor_ens]), cfg)
    regra = regra_selecao(r_lin["mae_dobras"], r_ens["mae_dobras"], cfg.margem_equivalencia)
    selecionado = melhor_ens if regra["ensemble_selecionado"] else LINEAR
    return dict(melhor_ensemble=melhor_ens, selecionado=selecionado, cv_MAE_linear=r_lin["MAE"],
                cv_MAE_ensemble=r_ens["MAE"], **{k: v for k, v in regra.items()})


def comparar_no_teste(y, p_a, p_b, n_boot: int, seed: int) -> dict:
    """Descritivo no teste: diferença de MAE (b − a) com IC bootstrap pareado + Wilcoxon."""
    ea, eb = np.abs(np.asarray(p_a) - y), np.abs(np.asarray(p_b) - y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        bs.append(eb[i].mean() - ea[i].mean())
    return {"dif_MAE_(b−a)": float((eb - ea).mean()), "IC95_inf": float(np.percentile(bs, 2.5)),
            "IC95_sup": float(np.percentile(bs, 97.5)), "p_Wilcoxon": float(stats.wilcoxon(ea, eb).pvalue)}


# ------------------------------------------------------------------ perda de quantil (pinball) — pedido do avaliador (p. 17)
def competir_quantilica(base, meta_sel, alvo, trilha, idx_tr, idx_te, params: dict, cfg: Config, quantis=QUANTIS):
    """Mean pinball loss em vários τ — usa TODA a amostra (não reduz aos extremos). Baseline = quantil
    empírico do treino. Árvores reaproveitam os hiperparâmetros vencedores da trilha."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import QuantileRegressor
    from xgboost import XGBRegressor

    Xtr, Xte = X_de(base.loc[idx_tr], meta_sel), X_de(base.loc[idx_te], meta_sel)
    ytr, yte = base.loc[idx_tr, alvo].values, base.loc[idx_te, alvo].values
    pre_l, pre_t = montar_preprocessador(meta_sel, True), montar_preprocessador(meta_sel, False)
    Ztr, Zte = pre_l.fit_transform(Xtr), pre_l.transform(Xte)
    Ztr2, Zte2 = pre_t.fit_transform(Xtr), pre_t.transform(Xte)
    linhas = []
    for tau in quantis:
        row = {"quantil τ": tau,
               "Baseline (quantil empírico)": mean_pinball_loss(yte, np.full(len(yte), np.quantile(ytr, tau)), alpha=tau)}
        melhor = None
        for al in (0.002, 0.005, 0.02, 0.08):
            e = []
            for a_, b_ in KFold(3, shuffle=True, random_state=cfg.seed).split(Ztr):
                q = QuantileRegressor(quantile=tau, alpha=al, solver="highs").fit(Ztr[a_], ytr[a_] - ytr[a_].mean())
                e.append(mean_pinball_loss(ytr[b_], q.predict(Ztr[b_]) + ytr[a_].mean(), alpha=tau))
            if melhor is None or np.mean(e) < melhor[0]:
                melhor = (np.mean(e), al)
        q = QuantileRegressor(quantile=tau, alpha=melhor[1], solver="highs").fit(Ztr, ytr - ytr.mean())
        row["Linear (regressão quantílica L1)"] = mean_pinball_loss(yte, _clip(trilha, q.predict(Zte) + ytr.mean()), alpha=tau)
        px = XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau, tree_method="hist", n_jobs=1,
                          random_state=cfg.seed, verbosity=0, **params["XGBoost"]).fit(Ztr2, ytr).predict(Zte2)
        row["XGBoost (quantile)"] = mean_pinball_loss(yte, _clip(trilha, px), alpha=tau)
        pl = LGBMRegressor(objective="quantile", alpha=tau, n_jobs=1, random_state=cfg.seed, verbose=-1,
                           **params["LightGBM"]).fit(Ztr2, ytr).predict(Zte2)
        row["LightGBM (quantile)"] = mean_pinball_loss(yte, _clip(trilha, pl), alpha=tau)
        linhas.append(row)
    df = pd.DataFrame(linhas).set_index("quantil τ")
    m = df.mean().rename("média dos quantis")
    g = (100 * (1 - m / m["Baseline (quantil empírico)"])).rename("ganho vs baseline (%)")
    return pd.concat([df, pd.DataFrame([m, g])])
