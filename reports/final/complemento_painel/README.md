# Complemento: agregados do painel (protótipo)

Tabelas que alimentam o painel em `docs/painel/index.html`. Só agregados: desfechos por UF, região, localização e
faixa de INSE, sem e com peso amostral, e o resíduo médio do modelo selecionado (observado − previsto fora da
amostra, validação cruzada 5 × 5). Grupos com menos de 5 escolas teriam os valores suprimidos. Nesta execução,
nenhum grupo ficou abaixo desse limite (a menor UF tem 12 escolas).

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `2f67e25` |
| Duração | 7,6 min |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
uv run python -m saeb_alfab.pipeline --modo completo --apenas painel
uv run python -m saeb_alfab.painel reports/final/complemento_painel/tabelas reports/final/tabelas docs/painel/index.html
```
