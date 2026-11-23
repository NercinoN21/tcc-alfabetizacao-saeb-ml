# Resultados finais (execução congelada)

Esta pasta contém a execução completa citada na monografia. Todo número do texto deve ser rastreável a uma
tabela daqui.

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `4dc8262` |
| Data da execução | 21/11/2026 (duração: 25,7 min) |
| Escolas (população principal) | 604 públicas, com INSE e pelo menos 10 alunos avaliados |
| Partição | 483 escolas de treino, 121 de teste |
| Variáveis candidatas | 363 (cenário A, sem cor/raça) |
| Dados | Saeb 2023 (Inep), conferidos pelos checksums em `data/raw/SHA256SUMS` |

Detalhes completos (versões dos pacotes, parâmetros, checksums) em `run_info.json`.

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout 4dc8262
make completo
```

## Organização das tabelas

| Prefixo | Conteúdo |
|---|---|
| `00_` | dicionário de variáveis: tipo, escala, mapa de codificação, tema |
| `01_` | funil de cobertura, cobertura dos blocos e decisão sobre cada variável. O diagnóstico dos pesos amostrais está em `complemento_pesos/` |
| `02_` | comparação de modelos e seleção do modelo final. A perda de quantil (pinball) está em `complemento_pinball/` |
| `03_` | explicabilidade: SHAP por variável e por tema, estabilidade e comparação LIME × SHAP. O ranking de temas sem × com peso amostral está em `complemento_ranking_pesos/` |
| `04_` | valor incremental dos blocos (ablação), robustez, sensibilidades, proxies e equidade |

Cada tabela existe em `.csv` (fonte) e `.tex` (para a monografia). As figuras estão em `figuras/`.
