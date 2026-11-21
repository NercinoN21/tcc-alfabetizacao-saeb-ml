"""Modelos (sem vazamento, predições válidas, determinismo) e avaliação (sMAPE, regra de seleção)."""
import numpy as np
import pandas as pd
import pytest

from saeb_alfab.avaliacao import regra_selecao, smape
from saeb_alfab.config import COLUNAS_PROIBIDAS_EM_X
from saeb_alfab.modelos import FractionalLogit, X_de, montar_preprocessador, pipeline_de, selecionar_variaveis


@pytest.fixture
def dados():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"A": rng.normal(size=n), "B": rng.normal(size=n), "C": rng.choice(["x", "y", "z"], n),
                       "GEO_ID_REGIAO": rng.choice(["1", "2"], n), "N_ALUNOS": rng.integers(10, 40, n)})
    df["Y_PROP_ALFAB"] = 1 / (1 + np.exp(-(0.8 * df.A - 0.5 * df.B)))
    df.loc[:5, "A"] = np.nan
    meta = pd.DataFrame({"variavel": ["A", "B", "C", "Y_PROP_ALFAB"], "tipo": ["numerico", "numerico", "nominal", "numerico"],
                         "excluir_sempre": False, "sensivel": False, "percepcao_desfecho": False, "tema": "t"})
    return df, meta


def test_fractional_logit_preve_em_0_1(dados):
    df, meta = dados
    ms = meta[meta.variavel.isin(["A", "B", "C"])]
    pre = montar_preprocessador(ms, True)
    Z = pre.fit_transform(X_de(df, ms))
    p = FractionalLogit(C=1.0).fit(Z, df.Y_PROP_ALFAB, sample_weight=df.N_ALUNOS).predict(Z)
    assert np.all((p > 0) & (p < 1))
    with pytest.raises(ValueError):
        FractionalLogit().fit(Z, df.Y_PROP_ALFAB + 1)


def test_coluna_de_alvo_nunca_entra_em_X(dados):
    df, meta = dados
    with pytest.raises(AssertionError):
        selecionar_variaveis(df, meta)                 # Y_PROP_ALFAB está no meta -> deve falhar
    assert "Y_PROP_ALFAB" in COLUNAS_PROIBIDAS_EM_X


def test_pipeline_deterministico_e_robusto_a_categoria_nova(dados):
    df, meta = dados
    ms = meta[meta.variavel.isin(["A", "B", "C"])]
    tr, te = df.iloc[:90], df.iloc[90:].copy()
    te.loc[te.index[0], "C"] = "categoria_inedita"
    preds = []
    for _ in range(2):
        pipe = pipeline_de("prop", "LightGBM", ms, seed=7, params={"n_estimators": 50}, n_arvores=50)
        pipe.fit(X_de(tr, ms), tr.Y_PROP_ALFAB)
        preds.append(pipe.predict(X_de(te, ms)))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_smape_valores_conhecidos():
    assert smape([100, 200], [100, 200]) == 0
    assert np.isclose(smape([100], [50]), 100 * 2 * 50 / 150)
    assert np.isclose(smape([0, 100], [0, 100]), 0)          # par 0/0 ignorado


def test_regra_de_selecao():
    rng = np.random.default_rng(3)
    lin = 0.160 + rng.normal(0, 0.004, 25)
    # ensemble indistinguível -> linear
    assert not regra_selecao(lin, lin - 0.0002 + rng.normal(0, 0.004, 25), margem=0.02)["ensemble_selecionado"]
    # ensemble claramente melhor (≈ 10%) e consistente -> ensemble
    assert regra_selecao(lin, lin - 0.016 + rng.normal(0, 0.0005, 25), margem=0.02)["ensemble_selecionado"]
    # melhor de forma significativa, mas abaixo da margem prática -> linear
    assert not regra_selecao(lin, lin - 0.0016 + rng.normal(0, 0.00005, 25), margem=0.02)["ensemble_selecionado"]


def test_lime_regressao_usa_o_sinal_correto():
    """Regressão: em modo 'regression' o LIME devolve as chaves 0 (espelho negativo) e 1 (sinal correto)."""
    from lime.lime_tabular import LimeTabularExplainer

    from saeb_alfab.explicabilidade import _lime_por_variavel

    X = np.random.default_rng(0).normal(size=(300, 3))
    f = lambda Z: 2 * Z[:, 0] - Z[:, 1]
    expl = LimeTabularExplainer(X, mode="regression", random_state=0)
    # Instância com contribuições verdadeiras de sinais opostos: a alto (contribui +), b alto (contribui −).
    x = np.array([1.5, 1.5, 0.0])
    pesos, _ = _lime_por_variavel(expl, x, f, 3, ["a", "b", "c"], 2000)
    assert pesos["a"] > 0 > pesos["b"]                 # com a chave 0 (espelho), os dois sinais inverteriam
