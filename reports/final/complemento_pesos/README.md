# Complemento: diagnóstico dos pesos amostrais

Diagnóstico do peso amostral da escola (`VL_PESO_ESCOLA`, do `TS_DIRETOR`) na população principal, usado na
Seção 3.3 da monografia. As sensibilidades de treino e avaliação com peso estão na execução principal
(`reports/final/tabelas/04_sensibilidades_*`, eixo "Pesos").

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `d245c8e` (as duas primeiras tabelas são idênticas às da versão `bdbe095`) |
| Data da execução | 24/11/2026 |
| Base | idêntica à da execução principal (604 escolas, 363 variáveis) |

## Tabelas

| Arquivo | Conteúdo |
|---|---|
| `01_pesos_diagnostico` | escolas, pesos imputados, coeficiente de variação, deff de Kish = n·Σw²/(Σw)², escolas efetivas e razões entre pesos |
| `01_pesos_medias` | médias com e sem peso dos dois desfechos e do INSE, diferença relativa e Spearman entre peso e variável |
| `01_alfabetizacao_por_rede` | proporção de alunos do 2º ano no corte de 743 pontos ou acima, por rede, com e sem o peso do aluno (`PESO_ALUNO_LP`), entre os alunos com proficiência calculada. Número da Introdução (49,3% na rede pública) |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout d245c8e
uv run python -m saeb_alfab.pipeline --modo completo --apenas pesos
```

## Acréscimo: intervalo da proporção por rede (commit `325340e`)

`01_alfabetizacao_por_rede_ic` traz o IC95% aproximado da proporção ponderada de alfabetizados por rede, por
bootstrap de escolas (2.000 reamostragens, semente 2110). O intervalo ignora a estratificação do desenho e tende
a ser um pouco mais largo. Rede pública: 49,3% (46,2% a 52,3%). Proveniência em `run_info_ic.json`. As demais
tabelas desta nova execução saíram idênticas às congeladas aqui.
