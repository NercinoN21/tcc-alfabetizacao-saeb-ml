# Complemento: ranking de temas entre as execuções de robustez

A Seção 3.9 privilegia a estabilidade do ordenamento dos temas entre execuções. Para cada execução, o modelo
selecionado (linear nas duas trilhas) é ajustado em toda a base da execução, com os mesmos hiperparâmetros, e a
participação de cada tema no |SHAP| é comparada à da referência (R1).

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `841bcd7` |
| Data da execução | 24/11/2026 (duração: 7,6 min) |
| Base | idêntica à da execução principal |

O Spearman usa os temas presentes nas duas execuções comparadas e exclui o INSE, que não existe em R2 e R3.

## Tabelas (por trilha: `prop` e `prof`)

| Arquivo | Conteúdo |
|---|---|
| `04_ranking_temas_execucoes_<trilha>` | participação de cada tema no \|SHAP\| em cada execução (R1, R2, R3, turma, n ≥ 20, junção interna) |
| `04_ranking_temas_execucoes_resumo_<trilha>` | temas comuns, Spearman com R1, temas em comum no top-5 e 1º tema sem o INSE |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout 841bcd7
uv run python -m saeb_alfab.pipeline --modo completo --apenas ranking-execucoes
```
