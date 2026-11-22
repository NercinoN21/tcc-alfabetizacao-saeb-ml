# Complemento: perda de quantil (pinball)

Esta pasta substitui as tabelas `02_pinball_*` da execução principal (`reports/final/`). Na versão anterior, a
regressão quantílica linear usava uma grade de penalidades grossa demais e terminava sem nenhuma variável em
três das quatro pontas, e as árvores reaproveitavam os hiperparâmetros escolhidos para o MAE. Agora cada modelo é
calibrado no treino para cada quantil, a trilha de proporção usa o peso de precisão n em todos os modelos e as
diferenças em relação ao baseline têm IC bootstrap pareado.

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `ec794be` |
| Data da execução | 23/11/2026 (duração: 3,3 min) |
| Base e partição | idênticas às da execução principal (604 escolas, 483 de treino, 121 de teste, 363 variáveis) |

## Tabelas (por trilha: `prop` e `prof`)

| Arquivo | Conteúdo |
|---|---|
| `02_pinball_<trilha>` | perda pinball média no teste por quantil τ, média dos quantis e ganho sobre o baseline |
| `02_pinball_ic_<trilha>` | diferença modelo − baseline por τ, com IC95% bootstrap pareado (negativo = modelo melhor) |
| `02_pinball_diagnostico_<trilha>` | penalidade L1 escolhida, coeficientes não nulos e perda na CV do treino, por τ |
| `02_pinball_cruzamento_<trilha>` | fração das escolas de teste com quantis previstos fora de ordem |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout ec794be
uv run python -m saeb_alfab.pipeline --modo completo --apenas pinball
```
