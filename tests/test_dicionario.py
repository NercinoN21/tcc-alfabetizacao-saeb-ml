"""Regressão do erro de codificação do TCC 1: tipo pelo dicionário e escalas pelo texto, não pela letra."""
import numpy as np
import pandas as pd
import pytest

from saeb_alfab import dicionario as D

from .conftest import DADOS_REAIS


def _meta(dic_prof, dic_dir):
    return D.montar_meta(D.parse_dicionario(dic_prof), D.parse_dicionario(dic_dir)).set_index("variavel")


def test_itens_NUM_sao_numericos_mesmo_sem_slidebar(dic_prof, dic_dir):
    m = _meta(dic_prof, dic_dir)
    for v in ("PROF_TX_Q002", "PROF_TX_Q048", "DIR_TX_Q017"):
        assert m.loc[v, "tipo"] == "numerico", v


def test_escala_mista_separa_fora_de_escala_em_indicador(dic_prof, dic_dir):
    m = _meta(dic_prof, dic_dir).reset_index()
    linha = m.set_index("variavel").loc["PROF_TX_Q069"].copy()
    linha["variavel"] = "PROF_TX_Q069"
    s = pd.Series(["A", "B", "C", "D", "E", "F", None])
    out = D.codificar_item(s, linha)
    np.testing.assert_array_equal(out["PROF_TX_Q069"].values, [np.nan, np.nan, 1, 2, 3, 4, np.nan])
    np.testing.assert_array_equal(out["PROF_TX_Q069__FORA_ESCALA"].values, [1, 1, 0, 0, 0, 0, np.nan])


def test_ordinal_decrescente_e_mapeado_pelo_texto(dic_prof, dic_dir):
    m = _meta(dic_prof, dic_dir)
    assert m.loc["PROF_TX_Q033", "tipo"] == "ordinal"
    assert m.loc["PROF_TX_Q033", "mapa"] == {"A": 3.0, "B": 2.0, "C": 1.0}      # 'todas' > 'algumas' > 'nenhuma'
    assert m.loc["PROF_TX_Q021", "mapa"] == {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}


def test_classificacoes_especiais(dic_prof, dic_dir):
    m = _meta(dic_prof, dic_dir)
    assert m.loc["PROF_TX_Q050", "tipo"] == "binario" and m.loc["PROF_TX_Q050", "mapa"] == {"A": 0.0, "B": 1.0}
    assert m.loc["DIR_TX_Q101", "escala"] == "sim_nao+fora"                       # 'Não sei' -> indicador
    assert m.loc["PROF_TX_Q109", "tipo"] == "nominal"
    assert bool(m.loc["PROF_TX_Q148", "excluir_sempre"])                          # resposta aberta
    assert m.loc["DIR_TX_Q015_B", "tipo"] == "binario"                            # marcação única


@pytest.mark.dados
@pytest.mark.skipif(not DADOS_REAIS, reason="microdados ausentes")
def test_dicionario_real_sem_itens_sem_classificacao():
    from saeb_alfab.config import Config

    pl = pd.read_excel(Config().data_dir / "Dicionario_Saeb_2023.xlsx", sheet_name=None, header=None)
    m = D.montar_meta(D.parse_dicionario(pl["TS_PROFESSOR"]), D.parse_dicionario(pl["TS_DIRETOR"])).set_index("variavel")
    assert (m.tipo == "numerico").sum() >= 23
    assert m.loc["DIR_TX_Q017", "tipo"] == "numerico"
    assert m.loc["PROF_TX_Q069", "escala"] == "adequacao+fora"
    assert set(m.tipo) <= {"numerico", "ordinal", "binario", "nominal", "texto"}


@pytest.mark.dados
@pytest.mark.skipif(not DADOS_REAIS, reason="microdados ausentes")
def test_verificacao_exaustiva_dicionario_x_codigo_x_dados():
    """Todo item: (1) letras dos CSVs existem no dicionário; (2) ordinais seguem a ordem do TEXTO da escala,
    com opções fora da escala marcadas; (3) binárias têm o polo positivo = 1; (4) numéricos dentro da faixa."""
    import re

    from saeb_alfab.config import Config
    from saeb_alfab.etl import CSV_KW

    pasta = Config().data_dir
    pl = pd.read_excel(pasta / "Dicionario_Saeb_2023.xlsx", sheet_name=None, header=None)
    problemas = []
    for pref, aba, csv in (("PROF", "TS_PROFESSOR", "TS_PROFESSOR.csv"), ("DIR", "TS_DIRETOR", "TS_DIRETOR.csv")):
        dic = D.parse_dicionario(pl[aba])
        meta = D.montar_meta(dic, dic.iloc[:0]) if pref == "PROF" else D.montar_meta(dic.iloc[:0], dic)
        meta = meta.set_index("variavel")
        raw = pd.read_csv(pasta / csv, **CSV_KW)
        for var, g in dic.groupby("var", sort=False):
            nome = f"{pref}_{var}"
            if nome not in meta.index or var not in raw:
                continue
            m, s = meta.loc[nome], raw[var]
            if m.tipo == "numerico":
                lim = [int(x) for x in re.findall(r"\d+", str(g.resp.iloc[0]))]
                v = pd.to_numeric(s, errors="coerce")
                if len(lim) == 2 and ((v < lim[0]) | (v > lim[1])).any():
                    problemas.append((nome, "fora da faixa"))
                continue
            ops = [(str(c).strip().upper(), D.norm(r)) for c, r in zip(g.cod, g.resp)]
            letras = set(s.dropna().astype(str).str.strip().str.upper())
            if m.tipo != "texto" and letras - {c for c, _ in ops} and len(ops) > 1:
                problemas.append((nome, "letra sem correspondência", letras - {c for c, _ in ops}))
            mapa = m.mapa if isinstance(m.mapa, dict) else {}
            if m.tipo == "ordinal" and m.escala != "faixa_salarial":
                esc = D.ESCALAS[m.escala.replace("+fora", "")]
                for c, t in ops:
                    esperado = "FORA" if t in D.FORA_DE_ESCALA else esc.index(t) + (0 if m.escala.startswith("contagem") else 1)
                    if mapa.get(c) != esperado:
                        problemas.append((nome, c, t, mapa.get(c), esperado))
            if m.tipo == "binario" and len(ops) == 2 and m.escala and m.escala.startswith("sim_nao"):
                for c, t in ops:
                    if t in D.POSITIVOS_BINARIOS and mapa.get(c) != 1.0:
                        problemas.append((nome, "polo positivo ≠ 1", c, t))
    assert not problemas, problemas[:10]
