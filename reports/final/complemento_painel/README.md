# Complemento: agregados do painel (protótipo)

Tabelas que alimentam o painel em `docs/painel/index.html`. Só agregados: desfechos por UF, região, localização e
faixa de INSE, sem e com peso amostral, e o resíduo médio do modelo selecionado (observado − previsto fora da
amostra, validação cruzada 5 × 5). Grupos com menos de 5 escolas teriam os valores suprimidos. Nesta execução,
nenhum grupo ficou abaixo desse limite (a menor UF tem 12 escolas).

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `325340e` (antes `2f67e25`) |
| Duração | 7,6 min |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
uv run python -m saeb_alfab.pipeline --modo completo --apenas painel
uv run python -m saeb_alfab.painel reports/final/complemento_painel/tabelas reports/final/tabelas docs/painel/index.html
```

## Atualização (commit `325340e`)

As tabelas ganharam o IC95% das médias com peso (`IC inf`, `IC sup`), por bootstrap das escolas de cada grupo
(2.000 reamostragens, semente 2110). As colunas que já existiam diferem da versão anterior em menos de 1e-5, efeito
numérico do reajuste do modelo, sem mudança em nenhum valor exibido.
