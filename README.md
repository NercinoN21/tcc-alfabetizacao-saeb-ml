# Preditores da alfabetização e da proficiência em Língua Portuguesa

**Uma comparação de algoritmos aplicada ao 2º ano do Ensino Fundamental público (Saeb 2023)**
Trabalho de Conclusão de Curso · Ciência de Dados para Negócios · UFPB
Autor: Nercino José De Barros Neto · Orientador: Prof. Dr. Alessio Tony Cavalcanti De Almeida

Base analítica no nível da **escola** (escolas públicas com 2º ano avaliado no Saeb 2023), construída a partir dos
microdados de aluno, escola, professor e diretor. Pergunta: quanto os questionários contextuais de professor e
diretor acrescentam à previsão da **proporção de alunos alfabetizados** e da **proficiência média em Língua
Portuguesa**, além do nível socioeconômico (INSE) e da geografia? Modelos lineares regularizados e *ensembles* de
árvores são comparados sob regra de seleção definida antes da estimação; o modelo selecionado é explicado com SHAP
e LIME. Os resultados são **associativos** e válidos no **nível da escola**.

## Reproduzir

```bash
uv sync          # Python 3.12 e versões fixadas (uv.lock)
make dados       # baixa do Inep só os arquivos necessários e confere os checksums
make testes      # testes automatizados
make completo    # execução completa → reports/execucoes/
```

Os dados usados no trabalho estão congelados em `data/snapshot/` (22 MB, os arquivos originais do Inep apenas
comprimidos): `make dados-snapshot` os restaura e confere os checksums. Alternativamente, `make dados` baixa do Inep
(o Inep republica os arquivos; os checksums indicam se a versão mudou). Os resultados citados na monografia estão em
`reports/final/`.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `src/saeb_alfab/` | pipeline: extração, dicionário, base por escola, modelos, avaliação, explicabilidade, robustez |
| `tests/` | testes automatizados (pytest) |
| `monografia/` | texto da monografia em LaTeX (abnTeX2, modelo UFPB) |
| `data/snapshot/` | cópia congelada dos microdados usados (Inep, CC BY 4.0) |
| `docs/` | notas metodológicas (ex.: espaços de busca de hiperparâmetros) |
| `reports/final/` | tabelas e figuras da execução citada no texto |

## Dados

Microdados do Saeb 2023 — Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).
Fonte: <https://download.inep.gov.br/microdados/microdados_saeb_2023.zip>.

## Licenças

- **Código** (`src/`, `tests/`, `Makefile`): [MIT](LICENSE).
- **Dados** (`data/snapshot/`): Inep, Creative Commons Atribuição 4.0 (CC BY 4.0), Portaria Inep nº 496/2021.
- **Texto da monografia** (`monografia/`): [CC BY 4.0](LICENSE-MONOGRAFIA.md) — reuso livre com crédito ao autor.

Para citar, use o arquivo [`CITATION.cff`](CITATION.cff).
