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


def test_itens_de_percepcao_do_desfecho_fora_por_padrao():
    """D5: PROF_TX_Q080 e DIR_TX_Q030 saem da especificação principal; a sensibilidade os reinclui."""
    from saeb_alfab.config import Config

    assert Config().excluir_percepcao_desfecho is True
    n = 50
    df = pd.DataFrame({"PROF_TX_Q080": np.arange(n) % 4 + 1.0, "PROF_TX_Q081": np.arange(n) % 3 + 1.0})
    meta = pd.DataFrame({"variavel": ["PROF_TX_Q080", "PROF_TX_Q081"], "tipo": "ordinal", "excluir_sempre": False,
                         "sensivel": False, "percepcao_desfecho": [True, False], "tema": "t"})
    principal, _ = selecionar_variaveis(df, meta, excluir_percepcao=True)
    sensibilidade, _ = selecionar_variaveis(df, meta, excluir_percepcao=False)
    assert principal.variavel.tolist() == ["PROF_TX_Q081"]
    assert set(sensibilidade.variavel) == {"PROF_TX_Q080", "PROF_TX_Q081"}


def test_perda_pinball_valores_conhecidos():
    from saeb_alfab.avaliacao import perda_pinball
    # u = y − q: acima do quantil pesa τ, abaixo pesa (1 − τ)
    assert np.allclose(perda_pinball([10, 10], [8, 12], 0.9), [0.9 * 2, 0.1 * 2])
    assert np.allclose(perda_pinball([1.0], [1.0], 0.1), [0.0])


def test_ic_pinball_pareado_detecta_modelo_melhor():
    from saeb_alfab.avaliacao import BASE_QUANTIL, ic_bootstrap_pinball
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    quantis = (0.1, 0.5, 0.9)
    from scipy.stats import norm
    preds = {BASE_QUANTIL: {t: np.full(300, 0.0) for t in quantis},
             "bom": {t: np.full(300, norm.ppf(t)) for t in quantis}}
    ic = ic_bootstrap_pinball(y, preds, quantis, n_boot=300, seed=1)
    media = ic.loc[("bom", "média dos quantis")]
    assert media["IC95_sup"] < 0 and media["ganho_vs_baseline_%"] > 0   # o quantil verdadeiro vence a constante 0
    assert len(ic) == len(quantis) + 1


def test_competir_quantilica_linear_nao_colapsa_com_sinal_forte(dados):
    """Regressão do bug da grade de penalidades: com sinal forte, a regressão quantílica L1 precisa
    manter coeficientes nas pontas (antes, a grade grossa podia levá-la ao só-intercepto)."""
    from dataclasses import replace
    from saeb_alfab.avaliacao import competir_quantilica
    from saeb_alfab.config import Config
    df, meta = dados
    df = df.copy()
    df["Y_PROF_MEDIA"] = 750 + 30 * df.B + np.random.default_rng(2).normal(0, 3, len(df))
    ms = meta[meta.variavel.isin(["A", "B", "C"])]
    cfg = replace(Config(), modo="rapido")
    perdas, ic, diag, cruz = competir_quantilica(df, ms, "Y_PROF_MEDIA", "prof", df.index[:90], df.index[90:], cfg,
                                                 quantis=(0.1, 0.9))
    assert (diag["linear: coeficientes ≠ 0"] > 0).all()
    assert perdas.loc["ganho vs baseline (%)", "Linear (regressão quantílica L1)"] > 50
    assert set(cruz.index) == {"Linear (regressão quantílica L1)", "XGBoost (quantile)", "LightGBM (quantile)"}


def test_ranking_temas_ponderado_reflete_os_pesos():
    from saeb_alfab.explicabilidade import ranking_temas_ponderado
    # 3 temas. Sem peso, "a" domina. As escolas de peso alto têm |SHAP| concentrado em "c".
    abs_escola = pd.DataFrame({"x1": [5, 5, 5, 0.1], "x2": [1, 1, 1, 1], "x3": [0, 0, 0, 10]})
    meta = pd.DataFrame({"variavel": ["x1", "x2", "x3"], "tema": ["a", "b", "c"]})
    tab, resumo = ranking_temas_ponderado(abs_escola, meta, [1, 1, 1, 1], n_boot=50, seed=0)
    assert resumo.loc["Spearman sem × com peso", "valor"] == pytest.approx(1.0)       # pesos iguais: mesmo ranking
    tab, resumo = ranking_temas_ponderado(abs_escola, meta, [1, 1, 1, 50], n_boot=50, seed=0)
    assert tab.index[0] == "a"                                   # sem peso: a (15,1) > c (10)
    assert resumo.loc["1º tema com peso", "valor"] == "c"        # com peso, a escola pesada põe c na frente
    assert tab["participação com peso"].sum() == pytest.approx(1.0)
