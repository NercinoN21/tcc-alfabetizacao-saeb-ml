"""Configuração única do projeto: caminhos, semente, parâmetros do protocolo.

Toda constante metodológica mora aqui (fonte única de verdade). Variáveis de ambiente
permitem trocar o modo sem editar código:  TCC_MODO=rapido|completo, SAEB_DATA_DIR=<pasta>.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ constantes do domínio
CORTE_ALFAB = 743            # Inep (2023): alfabetizado <=> PROFICIENCIA_LP_SAEB >= 743
URL_INEP = "https://download.inep.gov.br/microdados/microdados_saeb_2023.zip"
ARQUIVOS_SAEB = ["TS_ALUNO_2EF.csv", "TS_ESCOLA.csv", "TS_PROFESSOR.csv", "TS_DIRETOR.csv",
                 "Dicionario_Saeb_2023.xlsx"]
QUANTIS = (0.10, 0.25, 0.50, 0.75, 0.90)

# Colunas que NUNCA podem entrar em X (vazamento de alvo, desenho amostral, identificadores).
COLUNAS_PROIBIDAS_EM_X = {
    "Y_PROP_ALFAB", "Y_PROF_MEDIA", "Y_PROF_MEDIANA", "PROFICIENCIA_LP_SAEB", "IN_ALFABETIZADO",
    "ALFAB", "PESO", "PESO_ESCOLA", "PESO_ALUNOS_SOMA", "PESO_ALUNO_MEDIO", "ESTRATO",
    "ID_ESCOLA", "ID_TURMA", "ID_MUNICIPIO", "N_ALUNOS", "N_TURMAS_ALUNOS", "PUBLICA",
    "TEM_PROF", "TEM_DIR", "TEM_INSE",
}


@dataclass(frozen=True)
class Config:
    """Parâmetros de uma execução. `modo="rapido"` serve só para validar o pipeline;
    nenhum número de execução rápida deve ser citado no texto."""

    modo: str = field(default_factory=lambda: os.environ.get("TCC_MODO", "completo"))
    seed: int = 2110
    n_min_alunos: int = 10
    juncao: str = "esquerda"             # "esquerda" (texto, Seção 3.1.2) | "interna"
    limite_ausencia: float = 0.30
    teste: float = 0.20
    # Regra de seleção do modelo final (Seção 3.7): o ensemble só é escolhido se for
    # melhor que o linear com IC95% excluindo zero E com ganho relativo >= margem.
    margem_equivalencia: float = 0.02
    # Itens que medem a percepção do respondente sobre o próprio desfecho (PROF_TX_Q080, DIR_TX_Q030;
    # risco de causalidade reversa): fora da especificação principal; a sensibilidade os reinclui.
    excluir_percepcao_desfecho: bool = True
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAEB_DATA_DIR", RAIZ / "data" / "raw")))
    saida_dir: Path = field(default_factory=lambda: RAIZ / "reports" / "execucoes")

    @property
    def rapido(self) -> bool:
        return self.modo == "rapido"

    @property
    def n_iter(self) -> int:            # combinações sorteadas por modelo na busca aleatória
        return 8 if self.rapido else 40

    @property
    def n_arvores_rf(self) -> int:
        return 150 if self.rapido else 500

    @property
    def repet_cv(self) -> int:          # repetições da CV 5-dobras (OOF, ablação, sensibilidades, seleção)
        return 2 if self.rapido else 5

    @property
    def repet_estab(self) -> int:       # repetições da estabilidade do ranking SHAP
        return 3 if self.rapido else 20

    @property
    def n_iter_quantil(self) -> int:    # busca aleatória por τ das árvores com perda de quantil
        return 3 if self.rapido else 15

    @property
    def n_boot(self) -> int:
        return 200 if self.rapido else 2000
