# Complemento: diagnóstico dos pesos amostrais

Diagnóstico do peso amostral da escola (`VL_PESO_ESCOLA`, do `TS_DIRETOR`) na população principal, usado na
Seção 3.3 da monografia. As sensibilidades de treino e avaliação com peso estão na execução principal
(`reports/final/tabelas/04_sensibilidades_*`, eixo "Pesos").

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `bdbe095` |
| Data da execução | 24/11/2026 |
| Base | idêntica à da execução principal (604 escolas, 363 variáveis) |

## Tabelas

| Arquivo | Conteúdo |
|---|---|
| `01_pesos_diagnostico` | escolas, pesos imputados, coeficiente de variação, deff de Kish = n·Σw²/(Σw)², escolas efetivas e razões entre pesos |
| `01_pesos_medias` | médias com e sem peso dos dois desfechos e do INSE, diferença relativa e Spearman entre peso e variável |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout bdbe095
uv run python -m saeb_alfab.pipeline --modo completo --apenas pesos
```
