"""Seleção de variáveis, pré-processamento, estimadores e espaços de busca (Seções 3.5–3.6).

Os limites dos espaços de busca estão justificados em `docs/hiperparametros.md`
(pedido do avaliador, p. 32). O pré-processamento é sempre ajustado DENTRO de cada dobra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import COLUNAS_PROIBIDAS_EM_X

LINEAR = "Linear (Elastic Net)"
ENSEMBLES = ["Random Forest", "XGBoost", "LightGBM"]
MODELOS = [LINEAR] + ENSEMBLES


# ------------------------------------------------------------------ seleção de variáveis
def selecionar_variaveis(base: pd.DataFrame, meta: pd.DataFrame, com_cor: bool = False, com_inse: bool = True,
                         limite_ausencia: float = 0.30, excluir_percepcao: bool = False):
    """Variáveis candidatas de um (população × cenário). Regra de >30% de ausência aplicada na
    população em uso (ausência = escola sem a informação). Devolve (selecionadas, decisões)."""
    m = meta[meta.variavel.isin(base.columns)].copy()
    m["ausencia"] = [base[v].isna().mean() for v in m.variavel]
    m["n_unicos"] = [base[v].nunique(dropna=True) for v in m.variavel]
    motivo = pd.Series("incluída", index=m.index)
    motivo[m.excluir_sempre] = "excluída: texto aberto / saúde / redundante"
    if not com_cor:
        motivo[m.sensivel] = "excluída no cenário A (cor/raça, sensível)"
    if not com_inse:
        motivo[m.variavel.isin(["ESC_INSE", "ESC_PC_FORMACAO_DOCENTE_INICIAL"])] = "excluída: variável de TS_ESCOLA"
    if excluir_percepcao:
        motivo[m.percepcao_desfecho & (motivo == "incluída")] = "excluída: percepção do desfecho"
    motivo[(motivo == "incluída") & (m.ausencia > limite_ausencia)] = f"excluída: >{int(limite_ausencia * 100)}% de ausência"
    motivo[(motivo == "incluída") & (m.n_unicos <= 1)] = "excluída: constante"
    m["decisao"] = motivo
    sel = m[m.decisao == "incluída"].copy()
    proibidas = set(sel.variavel) & COLUNAS_PROIBIDAS_EM_X
    if proibidas:
        raise AssertionError(f"colunas de vazamento selecionadas como preditoras: {proibidas}")
    return sel, m


def X_de(base: pd.DataFrame, meta_sel: pd.DataFrame) -> pd.DataFrame:
    X = base[meta_sel.variavel.tolist()].copy()
    for v in meta_sel[meta_sel.tipo == "nominal"].variavel:
        X[v] = X[v].astype(object).where(X[v].notna(), np.nan)
    return X


def montar_preprocessador(meta_sel: pd.DataFrame, escalar: bool) -> ColumnTransformer:
    """Imputação (mediana / 'não informado'), one-hot e (opcional) padronização."""
    num = meta_sel[meta_sel.tipo != "nominal"].variavel.tolist()
    nom = meta_sel[meta_sel.tipo == "nominal"].variavel.tolist()
    t = []
    if num:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
                                  + ([("sc", StandardScaler())] if escalar else [])), num))
    if nom:
        t.append(("nom", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="não informado")),
                                   ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                                        sparse_output=False))]), nom))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=True)


def variavel_original(nomes_saida, meta_sel: pd.DataFrame) -> list[str]:
    """Mapeia colunas pós-pré-processamento (num__X, nom__X_cat) para a variável original."""
    nominais = sorted(meta_sel[meta_sel.tipo == "nominal"].variavel, key=len, reverse=True)
    out = []
    for n in nomes_saida:
        if n.startswith("num__"):
            out.append(n[5:])
        else:
            out.append(next((v for v in nominais if n[5:].startswith(v)), n[5:]))
    return out


# ------------------------------------------------------------------ regressão fracionária logit
class FractionalLogit(BaseEstimator, RegressorMixin):
    """Quase-verossimilhança binomial com ligação logística para y ∈ [0, 1] (Papke; Wooldridge, 1996),
    com penalidade elastic-net. Cada escola vira duas linhas (y=1 com peso w·p; y=0 com peso w·(1−p)).
    Predições sempre em (0, 1)."""

    def __init__(self, C=1.0, l1_ratio=0.5, max_iter=5000):
        self.C, self.l1_ratio, self.max_iter = C, l1_ratio, max_iter

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("FractionalLogit exige y em [0, 1]")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.mean()
        Xd, yd = np.vstack([X, X]), np.r_[np.ones(len(y)), np.zeros(len(y))]
        wd = np.r_[w * y, w * (1 - y)]
        k = wd > 1e-12
        self.m_ = LogisticRegression(solver="saga", C=self.C, l1_ratio=self.l1_ratio,   # elastic-net via l1_ratio
                                     max_iter=self.max_iter, tol=1e-4)
        self.m_.fit(Xd[k], yd[k], sample_weight=wd[k])
        self.coef_, self.intercept_ = self.m_.coef_.ravel(), float(self.m_.intercept_[0])
        return self

    def predict(self, X):
        return self.m_.predict_proba(np.asarray(X, float))[:, 1]

    def decision_function(self, X):          # margem (log-odds), usada pelo SHAP linear
        return self.m_.decision_function(np.asarray(X, float))


# ------------------------------------------------------------------ especificação dos modelos
def especificar(trilha: str, seed: int, n_arvores: int = 500) -> dict:
    """nome -> (estimador, espaço de busca, padronizar?, usa peso de precisão n?).

    Comparação justa: o peso de precisão n (nº de alunos) é usado por
    TODOS os modelos da trilha de proporção, não só pelo fractional logit."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    prop = trilha == "prop"
    return {
        LINEAR: (
            FractionalLogit() if prop else ElasticNet(max_iter=50000, random_state=seed),
            ({"model__C": loguniform(1e-3, 10.0), "model__l1_ratio": uniform(0.05, 0.9)} if prop else
             {"model__alpha": loguniform(1e-3, 1e1), "model__l1_ratio": uniform(0.05, 0.9)}), True, prop),
        "Random Forest": (
            RandomForestRegressor(n_estimators=n_arvores, n_jobs=1, random_state=seed),
            {"model__max_features": [0.05, 0.1, 0.2, 0.33, 0.5, "sqrt"], "model__min_samples_leaf": randint(1, 21),
             "model__max_samples": uniform(0.5, 0.5), "model__max_depth": [None, 4, 8, 16]}, False, prop),
        "XGBoost": (
            XGBRegressor(objective="reg:logistic" if prop else "reg:squarederror", tree_method="hist",
                         n_jobs=1, random_state=seed, verbosity=0),
            {"model__n_estimators": randint(100, 1001), "model__learning_rate": loguniform(0.005, 0.2),
             "model__max_depth": randint(2, 7), "model__subsample": uniform(0.5, 0.5),
             "model__colsample_bytree": uniform(0.2, 0.8), "model__min_child_weight": loguniform(1, 50),
             "model__reg_lambda": loguniform(0.1, 50), "model__reg_alpha": loguniform(1e-3, 10)}, False, prop),
        "LightGBM": (
            LGBMRegressor(objective="cross_entropy" if prop else "regression", n_jobs=1, random_state=seed, verbose=-1),
            {"model__n_estimators": randint(100, 1001), "model__learning_rate": loguniform(0.005, 0.2),
             "model__num_leaves": randint(4, 32), "model__min_child_samples": randint(5, 41),
             "model__colsample_bytree": uniform(0.2, 0.8), "model__subsample": uniform(0.5, 0.5),
             "model__subsample_freq": [1], "model__reg_lambda": loguniform(0.1, 50)}, False, prop),
    }


def pipeline_de(trilha: str, nome: str, meta_sel: pd.DataFrame, seed: int, params: dict | None = None,
                n_arvores: int = 500) -> Pipeline:
    est, _, escalar, _ = especificar(trilha, seed, n_arvores)[nome]
    if params:
        est.set_params(**params)
    return Pipeline([("pre", montar_preprocessador(meta_sel, escalar)), ("model", est)])


def usa_peso_n(trilha: str, nome: str) -> bool:
    return especificar(trilha, 0)[nome][3]


def particao(base: pd.DataFrame, alvo: str, seed: int, teste: float = 0.20):
    """80/20 por escola, estratificada por quintil do desfecho × região (categorias raras fundidas)."""
    q = pd.qcut(base[alvo].rank(method="first"), 5, labels=False).astype(str)
    chave = q + "_" + base["GEO_ID_REGIAO"].astype(str)
    chave = chave.where(chave.map(chave.value_counts()) >= 2, other=q)
    return train_test_split(base.index, test_size=teste, random_state=seed, stratify=chave)
