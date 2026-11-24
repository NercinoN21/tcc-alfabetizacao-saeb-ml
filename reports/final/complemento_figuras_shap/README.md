# Complemento: figuras da explicabilidade (SHAP e SHAP × LIME)

Figuras do modelo selecionado em cada trilha (linear nas duas), geradas por `saeb_alfab/figuras_shap.py`.

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `a723cfa` |
| Data da execução | 24/11/2026 (duração: 8,2 min) |
| Base, partição e modelo | idênticos aos da execução principal; o modelo selecionado foi reproduzido com a mesma semente |

A tabela `03_lime_vs_shap_<trilha>` reproduz a da execução principal (mesmas escolas e mesmos três primeiros
fatores; diferença numérica máxima de 0,0000014 na proporção, efeito do otimizador iterativo).

## Figuras

| Arquivo | Conteúdo |
|---|---|
| `03_shap_resumo_<trilha>.png` | contribuições SHAP nas 121 escolas de teste para as 12 variáveis principais; cor = valor da variável |
| `03_shap_lime_shap_<trilha>.png` | SHAP × LIME nos três casos escolhidos pelo autor (típica, acima e abaixo do esperado), como participação com sinal de cada variável no total de cada método |

Sem gráficos de dependência: com o modelo linear selecionado, a dependência de cada variável é uma reta
(coeficiente × valor) e nada acrescenta ao resumo. O código os gera quando o selecionado é um ensemble.

Na trilha de proporção, o SHAP está na escala da margem do modelo (log-odds) e o LIME na da proporção prevista.
Por isso a comparação usa a participação com sinal, que é o que as medidas de concordância comparam.

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout a723cfa
uv run python -m saeb_alfab.pipeline --modo completo --apenas figuras-shap
```
