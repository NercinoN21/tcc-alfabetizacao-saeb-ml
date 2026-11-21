# Dados

Os microdados descomprimidos (214 MB) **não são versionados**. Há duas formas de obtê-los:

```bash
make dados-snapshot   # restaura a cópia congelada em data/snapshot/ (mesmos dados do trabalho; recomendado)
make dados       # baixa do Inep só TS_ALUNO_2EF, TS_ESCOLA, TS_PROFESSOR, TS_DIRETOR e o dicionário (HTTP Range)
make checksums   # confere contra data/raw/SHA256SUMS
```

- Fonte: <https://download.inep.gov.br/microdados/microdados_saeb_2023.zip> (Inep, Saeb 2023).
- `SHA256SUMS` registra os arquivos usados no TCC. **O Inep republicou o zip em 22/07/2026**; se os checksums
  divergirem, os dados mudaram: não misture versões, registre a nova versão antes de prosseguir. Para reproduzir exatamente o trabalho, use o snapshot.
- Também é possível apontar para outra pasta: `SAEB_DATA_DIR=/caminho uv run python -m saeb_alfab.pipeline`.
- `processed/` recebe artefatos intermediários (também fora do git).
- Licença dos dados: CC BY 4.0 (Inep, Portaria nº 496/2021). Ver `snapshot/README.md`.
