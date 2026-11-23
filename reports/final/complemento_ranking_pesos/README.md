# Complemento: ranking de temas do SHAP sem e com peso amostral

Contraste entre a participação de cada tema no |SHAP| do modelo selecionado sem peso e com o peso amostral
da escola (decisão D2; Solon, Haider e Wooldridge, 2015), citado na Seção 3.3 da monografia.

| Campo | Valor |
|---|---|
| Modo | completo (citável no texto) |
| Semente | 2110 |
| Commit do código | `10a9d94` |
| Data da execução | 24/11/2026 (duração: 7,3 min) |
| Base e partição | idênticas às da execução principal (604 escolas, 483 de treino, 121 de teste, 363 variáveis) |
| Modelo explicado | o selecionado pela regra da Seção 3.7, reproduzido com a mesma semente (linear nas duas trilhas) |

Os dois rankings são calculados nas 604 escolas, com o modelo ajustado no treino: as 121 escolas de teste
sozinhas equivalem a poucas escolas efetivas quando ponderadas. Checagem: o ranking sem peso nas 604 escolas
tem Spearman de 0,996 (proporção) e 0,994 (proficiência) com o ranking oficial, calculado no teste
(`reports/final/tabelas/03_shap_temas_*`).

O IC do Spearman vem de bootstrap de escolas (percentis 2,5 e 97,5). Como a correlação está perto de 1, a
reamostragem tende a reduzi-la, e o limite superior pode ficar abaixo da estimativa pontual. O limite inferior
é o que importa para a leitura.

## Tabelas (por trilha: `prop` e `prof`)

| Arquivo | Conteúdo |
|---|---|
| `03_shap_temas_ponderado_<trilha>` | participação e posição de cada tema, sem e com peso |
| `03_shap_temas_ponderado_resumo_<trilha>` | escolas efetivas, Spearman entre os rankings com IC bootstrap, top-5 em comum, 1º tema e checagem contra o ranking oficial |

## Como reproduzir

```bash
uv sync --frozen
make dados-snapshot
git checkout 10a9d94
uv run python -m saeb_alfab.pipeline --modo completo --apenas ranking-pesos
```
