"""Integridade da base por escola (agregações e desfechos)."""
import numpy as np
import pytest

from saeb_alfab import base as B
from saeb_alfab import dicionario as D


@pytest.fixture
def construida(micro, dic_prof, dic_dir):
    aluno, escola, prof, dire = micro
    meta = D.montar_meta(D.parse_dicionario(dic_prof), D.parse_dicionario(dic_dir))
    return B.construir_base(aluno, escola, prof, dire, meta, "esquerda", 1), micro


def test_uma_linha_por_escola_e_aluno_sem_proficiencia_excluido(construida):
    (base, alunos, _, _), (aluno, *_ ) = construida
    assert base.index.is_unique and len(base) == 3
    assert len(alunos) == len(aluno) - 1
    assert base.N_ALUNOS.sum() == len(alunos)


def test_desfechos_corretos(construida):
    (base, alunos, _, _), _ = construida
    assert base.Y_PROP_ALFAB.between(0, 1).all()
    esperado = alunos.groupby("ID_ESCOLA").PROFICIENCIA_LP_SAEB.apply(lambda s: (s >= 743).mean())
    np.testing.assert_allclose(base.Y_PROP_ALFAB, esperado.reindex(base.index))
    np.testing.assert_allclose(base.Y_PROF_MEDIA, alunos.groupby("ID_ESCOLA").PROFICIENCIA_LP_SAEB.mean().reindex(base.index))


def test_agregacao_professor_por_tipo(construida):
    (base, _, prof_cod, _), _ = construida
    e1 = base.loc[1]
    assert e1.CTX_N_TURMAS == 2
    assert e1.PROF_TX_Q050 == 0.5                     # binária -> proporção de turmas com 'Sim'
    assert e1.PROF_TX_Q048 == 10.0                    # numérica -> média (5 e 15), não moda
    assert e1.PROF_TX_Q069__FORA_ESCALA == 0.5        # uma turma 'Não tem', outra 'Muito adequado'
    assert e1.PROF_TX_Q069 == 4.0                     # escala só com quem tem o recurso
    assert prof_cod.ID_TURMA.is_unique


def test_divergencia_do_corte_743_e_detectada(micro, dic_prof, dic_dir):
    aluno, escola, prof, dire = micro
    aluno = aluno.copy()
    aluno.loc[0, "IN_ALFABETIZADO"] = 1 - aluno.loc[0, "IN_ALFABETIZADO"]
    meta = D.montar_meta(D.parse_dicionario(dic_prof), D.parse_dicionario(dic_dir))
    with pytest.raises(AssertionError):
        B.construir_base(aluno, escola, prof, dire, meta)
