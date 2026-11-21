"""Fixtures sintéticas pequenas (os testes rápidos NÃO dependem dos microdados)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saeb_alfab.config import RAIZ


def _aba(linhas):
    """Simula uma aba do dicionário do Inep (lida com header=None): bloco de metadados + bloco de itens."""
    cab = ["Variável", "Tipo", "Tamanho", "Grupo", "Descrição", "Regra", "Código da Opção da Resposta", "Resposta"]
    topo = [["ID_ESCOLA", "NUM", 8, "", "id", "", "", ""], ["Variável", "", "", "", "", "", "", ""]]
    return pd.DataFrame(topo + [cab] + linhas)


@pytest.fixture
def dic_prof():
    L = [
        ["TX_Q002", "NUM", 2, "G", "Qual é a sua idade?", "", "slidebar", "18-70"],
        ["TX_Q048", "NUM", 2, "G", "Há quantos anos professor?", "", "", "0-30"],
        ["TX_Q021", "CHAR", 1, "G", "Quantos cursos?", "", "A", "Nenhuma."], [None, "CHAR", 1, "G", "", "", "B", "Uma."],
        [None, "CHAR", 1, "G", "", "", "C", "Duas."], [None, "CHAR", 1, "G", "", "", "D", "Três ou mais."],
        ["TX_Q050", "CHAR", 1, "G", "Trabalha em outra escola?", "", "A", "Não"], [None, "CHAR", 1, "G", "", "", "B", "Sim"],
        ["TX_Q069", "CHAR", 1, "G", "Livro didático", "", "A", "Não tem."], [None, "CHAR", 1, "G", "", "", "B", "Não uso."],
        [None, "CHAR", 1, "G", "", "", "C", "Muito inadequado."], [None, "CHAR", 1, "G", "", "", "D", "Inadequado."],
        [None, "CHAR", 1, "G", "", "", "E", "Adequado."], [None, "CHAR", 1, "G", "", "", "F", "Muito adequado."],
        ["TX_Q033", "CHAR", 1, "G", "Atividades", "", "A", "Todas as atividades formativas."],
        [None, "CHAR", 1, "G", "", "", "B", "Algumas atividades formativas (ou parte delas)."],
        [None, "CHAR", 1, "G", "", "", "C", "Nenhuma atividade formativa."],
        ["TX_Q109", "CHAR", 1, "G", "Quem decide?", "", "A", "Docente da turma."],
        [None, "CHAR", 1, "G", "", "", "B", "Equipe gestora."], [None, "CHAR", 1, "G", "", "", "C", "Outro."],
        ["TX_Q148", "CHAR", 1, "G", "Comentário", "", "", "Resposta aberta"],
    ]
    return _aba(L)


@pytest.fixture
def dic_dir():
    L = [
        ["TX_Q017", "NUM", 2, "G", "Há quantos anos diretor desta escola?", "", "", "0-30"],
        ["TX_Q015_B", "CHAR", 1, "G", "Nunca trabalhei", "", "A", "Nunca trabalhei."],
        ["TX_Q101", "CHAR", 1, "G", "Ocorreu?", "", "A", "Não"], [None, "CHAR", 1, "G", "", "", "B", "Sim"],
        [None, "CHAR", 1, "G", "", "", "C", "Não sei"],
    ]
    return _aba(L)


@pytest.fixture
def micro():
    """Microdados sintéticos: 3 escolas públicas, 4 turmas, alunos com proficiência conhecida."""
    rng = np.random.default_rng(0)
    linhas = []
    for esc, turmas in ((1, [11, 12]), (2, [21]), (3, [31])):
        for t in turmas:
            for k in range(12):
                prof = 700 + 10 * k + (40 if esc == 3 else 0)
                linhas.append(dict(ID_REGIAO=2, ID_UF=25, ID_MUNICIPIO=99, ID_AREA=2, ID_ESCOLA=esc, IN_PUBLICA=1,
                                   ID_LOCALIZACAO=1, ID_TURMA=t, IN_SITUACAO_CENSO=1, IN_PREENCHIMENTO_LP=1,
                                   IN_PROFICIENCIA_LP=1, IN_AMOSTRA=1, PESO_ALUNO_LP=rng.uniform(1, 3),
                                   PROFICIENCIA_LP_SAEB=float(prof), IN_ALFABETIZADO=int(prof >= 743)))
    aluno = pd.DataFrame(linhas)
    aluno.loc[len(aluno)] = {**aluno.iloc[0].to_dict(), "IN_PROFICIENCIA_LP": 0, "PROFICIENCIA_LP_SAEB": np.nan,
                             "IN_ALFABETIZADO": 0}                      # aluno sem proficiência: deve sair
    escola = pd.DataFrame({"ID_ESCOLA": [1, 2, 3], "IN_PUBLICA": 1,
                           "NIVEL_SOCIO_ECONOMICO": ["Nível II", "Nível IV", "Nível V"],
                           "PC_FORMACAO_DOCENTE_INICIAL": [50.0, 70.0, 90.0]})
    prof = pd.DataFrame({"ID_ESCOLA": [1, 1, 2, 3], "ID_TURMA": [11, 12, 21, 31], "ID_SERIE": 2, "SQ_QUESTIONARIO": 5,
                         "IN_PREENCHIMENTO_QUESTIONARIO": 1, "TX_Q002": [30, 50, 40, 45], "TX_Q048": [5, 15, 10, 20],
                         "TX_Q021": ["A", "D", "B", "C"], "TX_Q050": ["A", "B", "B", "A"],
                         "TX_Q069": ["A", "F", "E", "C"], "TX_Q033": ["A", "C", "B", "A"],
                         "TX_Q109": ["A", "A", "B", "C"], "TX_Q003": ["A", "C", "B", "A"], "TX_Q148": ["x", "", "", ""]})
    dire = pd.DataFrame({"ID_ESCOLA": [1, 2, 3], "ID_SERIE": 2, "IN_PREENCHIMENTO_QUESTIONARIO": 1,
                         "TX_Q017": [3.0, 10.0, 1.0], "TX_Q015_B": [None, "A", None], "TX_Q101": ["A", "C", "B"],
                         "TX_Q009": ["A", "C", "B"], "VL_PESO_ESCOLA": [2.0, 5.0, 1.0]})
    return aluno, escola, prof, dire


DADOS_REAIS = (RAIZ / "data" / "raw" / "TS_ALUNO_2EF.csv").exists()
