"""Integridade da base por escola (agregações e desfechos)."""
import numpy as np
import pandas as pd
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


def test_indicadores_de_ausencia_do_questionario(micro, dic_prof, dic_dir):
    """Ausência informativa: escola sem questionário do diretor recebe DIR_SEM_QUESTIONARIO = 1."""
    aluno, escola, prof, dire = micro
    meta = D.montar_meta(D.parse_dicionario(dic_prof), D.parse_dicionario(dic_dir))
    base, *_ = B.construir_base(aluno, escola, prof, dire[dire.ID_ESCOLA != 2], meta)
    assert base.loc[2, "DIR_SEM_QUESTIONARIO"] == 1.0 and base.loc[1, "DIR_SEM_QUESTIONARIO"] == 0.0
    assert (base.PROF_SEM_QUESTIONARIO == 0).all()
    assert set(base.PROF_SEM_QUESTIONARIO.unique()) <= {0.0, 1.0}


def test_deff_kish_valores_conhecidos():
    from saeb_alfab.robustez import deff_kish
    assert deff_kish([2.0, 2.0, 2.0]) == pytest.approx(1.0)            # pesos iguais: sem perda
    assert deff_kish([1, 1, 1, 3]) == pytest.approx(4 * 12 / 36)        # n·Σw²/(Σw)²


def test_diagnostico_pesos_contrasta_medias():
    from saeb_alfab.robustez import diagnostico_pesos
    pop = pd.DataFrame({"PESO": [1.0, 1.0, 2.0], "PESO_ESCOLA": [1.0, np.nan, 2.0],
                        "Y_PROP_ALFAB": [0.2, 0.4, 0.6], "Y_PROF_MEDIA": [700, 720, 760], "ESC_INSE": [3, 4, 5]})
    resumo, medias = diagnostico_pesos(pop)
    assert medias.loc["Y_PROP_ALFAB", "média ponderada"] == pytest.approx((0.2 + 0.4 + 1.2) / 4)
    assert resumo.loc["pesos imputados (sem questionário do diretor, %)", "valor"] == pytest.approx(100 / 3)


def test_alfabetizacao_por_rede_usa_peso_e_corte():
    from saeb_alfab.robustez import alfabetizacao_por_rede
    aluno = pd.DataFrame({"IN_PROFICIENCIA_LP": [1, 1, 1, 0], "IN_PUBLICA": [1, 1, 0, 1],
                          "PESO_ALUNO_LP": ["3,0", "1,0", "2,0", "9,0"],          # vírgula decimal, como no Inep
                          "PROFICIENCIA_LP_SAEB": ["750,5", "700,0", "743,0", "800,0"]})
    t = alfabetizacao_por_rede(aluno)
    assert t.loc["pública", "alunos"] == 2                                    # sem proficiência calculada fica fora
    assert t.loc["pública", "proporção com peso (%)"] == pytest.approx(75.0)  # 3/(3+1)
    assert t.loc["pública", "proporção sem peso (%)"] == pytest.approx(50.0)
    assert t.loc["privada", "proporção com peso (%)"] == pytest.approx(100.0) # 743 está no corte
