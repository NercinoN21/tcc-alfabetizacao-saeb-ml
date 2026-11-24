# Complemento: ablação com índices por tema

Repete a ablação da execução principal (`reports/final/tabelas/04_ablacao_*`) e acrescenta:

- a especificação **M7: INSE + geografia + índices por tema**, declarada na decisão D1 antes da execução
  final e implementada depois dela. Cada questionário (professor, diretor) é resumido em um índice por tema,
  o 1º componente principal das variáveis do tema, padronizadas, ajustado só no treino de cada dobra
  (`modelos.IndicesPorTema`): 44 índices (19 temas do professor, 25 do diretor);
- o contraste de cada especificação com **M1 (INSE + geografia)**, além do contraste com M4 (completo).

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `1a52ea8` |
| Data da execução | 24/11/2026 (duração: 8,7 min) |
| Base e partição | idênticas às da execução principal (604 escolas, 363 variáveis) |
| Modelo | o selecionado (linear nas duas trilhas), reproduzido com a mesma semente |

As especificações M0 a M6 e as retiradas de bloco reproduzem a execução principal: idênticas na proficiência
e com diferença máxima de 0,000003 no MAE da proporção, efeito do otimizador iterativo do *fractional logit*.

Δ MAE é a diferença nas mesmas dobras (5 × 5), com IC95% pelo t corrigido de Nadeau e Bengio (2003).
Contra M4, positivo = pior que o completo. Contra M1, negativo = melhor que INSE + geografia.

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout 1a52ea8
uv run python -m saeb_alfab.pipeline --modo completo --apenas ablacao
```
