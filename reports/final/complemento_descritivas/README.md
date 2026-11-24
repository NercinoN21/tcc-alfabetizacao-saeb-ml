# Complemento: análises descritivas

Análises descritivas da base por escola para o capítulo de Resultados, portadas do notebook do TCC 1 para o
pacote (`saeb_alfab/descritivas.py`). Toda média que descreve a rede sai sem peso e com o peso amostral da
escola (decisão D2, Seção 3.3).

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `0e37689` |
| Data da execução | 24/11/2026 |
| Base | idêntica à da execução principal (604 escolas, 363 variáveis) |

## Tabelas

| Arquivo | Conteúdo |
|---|---|
| `01_unidade_analise` | alunos, turmas, escolas, ICC da proficiência (escola e turma) e da alfabetização, % de escolas com mais de uma turma: coorte interna e população principal |
| `01_desfechos` | estatísticas dos desfechos, do nº de alunos e de turmas, com a média com peso |
| `01_desfechos_por_geografia` | por região, localização e área: escolas, alunos, INSE médio e desfechos sem e com peso |
| `01_desfechos_por_inse` | desfechos por nível de INSE, sem e com peso (os níveis I e VII têm uma escola cada) |
| `01_inse_sozinho` | Spearman entre INSE e desfecho e R² da média por nível de INSE |
| `01_funil_binomial` | % de escolas fora da banda de 95% do acaso binomial |
| `01_cobertura_por_variavel` | por item: não resposta, ausência na base por escola (regra dos 30%) e ausência ponderada |
| `01_cobertura_por_tema` | o mesmo, agregado por tema |
| `01_ausencia_informativa` | desfechos das escolas com e sem questionário de professor e de diretor (SMD e Mann-Whitney) |
| `01_vies_selecao_inse` | rede pública com n ≥ 10: escolas com e sem INSE (SMD e Mann-Whitney) |
| `01_cor_composicao` | % de professores e diretores pretos ou pardos por região, faixa de INSE e localização |
| `01_cor_associacao_bruta` | desfechos por cor do diretor e por composição racial dos professores |
| `01_cor_por_faixa_de_inse` | a mesma associação dentro de cada faixa de INSE |

## Figuras

| Arquivo | Conteúdo |
|---|---|
| `01_desfechos_distribuicao.png` | distribuição dos dois desfechos, com as médias sem e com peso e o corte de 743 |
| `01_desfechos_por_inse.png` | desfechos por faixa de INSE, sem e com peso, com IC 95% (escolas efetivas de Kish no ponderado) |
| `01_funil_binomial.png` | proporção de alfabetizados × alunos avaliados, com a banda de 95% do acaso binomial |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout 0e37689
uv run python -m saeb_alfab.pipeline --modo completo --apenas descritivas
```
