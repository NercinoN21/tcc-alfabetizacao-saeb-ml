# Snapshot dos dados usados no trabalho

`saeb2023_2ef_snapshot.tar.xz` contém, **sem nenhuma alteração de conteúdo**, os cinco arquivos do Saeb 2023 usados
neste trabalho: `TS_ALUNO_2EF.csv`, `TS_ESCOLA.csv`, `TS_PROFESSOR.csv`, `TS_DIRETOR.csv` e
`Dicionario_Saeb_2023.xlsx`. As únicas modificações são a seleção desses arquivos e a compressão (xz).

- **Fonte e autoria:** Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep), *Microdados do
  Saeb 2023*, <https://download.inep.gov.br/microdados/microdados_saeb_2023.zip>.
- **Licença dos dados:** Creative Commons Atribuição 4.0 (CC BY 4.0), licença padrão dos dados abertos do Inep
  (Portaria Inep nº 496, de 13 de outubro de 2021). Os microdados são publicados pelo Inep já anonimizados.
- **Por que existe:** o Inep republica os arquivos (o zip foi atualizado em 22/07/2026). O snapshot garante que os
  resultados possam ser reproduzidos com exatamente os mesmos dados.
- **Integridade:** `SHA256SUMS` (o arquivo comprimido) e `../raw/SHA256SUMS` (cada arquivo descomprimido).
- **Restaurar:** `make dados-snapshot` (confere os dois níveis de checksum).
