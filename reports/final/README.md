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
| `01_` | funil de cobertura, cobertura dos blocos e decisão sobre cada variável. O diagnóstico dos pesos amostrais está em `complemento_pesos/`, e as análises descritivas (desfechos, INSE, cobertura, viés de seleção, cor/raça), com figuras, em `complemento_descritivas/` |
| `02_` | comparação de modelos e seleção do modelo final. A perda de quantil (pinball) está em `complemento_pinball/` |
| `03_` | explicabilidade: SHAP por variável e por tema, estabilidade e comparação LIME × SHAP. O ranking de temas sem × com peso amostral está em `complemento_ranking_pesos/`, e as figuras (resumo do SHAP e SHAP × LIME dos casos do texto), em `complemento_figuras_shap/` |
| `04_` | valor incremental dos blocos (ablação), robustez, sensibilidades, proxies e equidade. A ablação com índices por tema e o contraste com INSE + geografia estão em `complemento_ablacao_indices/`, e o ranking de temas entre as execuções de robustez, em `complemento_ranking_execucoes/`, e os agregados do painel, em `complemento_painel/` |

Cada tabela existe em `.csv` (fonte) e `.tex` (para a monografia). As figuras estão em `figuras/`.
