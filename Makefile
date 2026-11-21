# Atalhos do projeto (requer uv: https://docs.astral.sh/uv/)
.PHONY: ambiente testes testes-dados rapido completo dados checksums dados-snapshot pdf

ambiente:        ## cria/atualiza o ambiente com as versões fixadas em uv.lock
	uv sync

testes:          ## testes rápidos (sintéticos, < 5 s)
	uv run pytest -q -m "not dados"

testes-dados:    ## todos os testes, incluindo os que leem data/raw
	uv run pytest -q

rapido:          ## valida o pipeline de ponta a ponta (números NÃO citáveis)
	uv run python -m saeb_alfab.pipeline --modo rapido

completo:        ## execução final (pode levar horas); depois copie a pasta para reports/final/
	uv run python -m saeb_alfab.pipeline --modo completo

dados:           ## baixa do Inep só os arquivos necessários e confere os checksums
	uv run python -c "from saeb_alfab import etl; from saeb_alfab.config import Config; c=Config(); print(etl.baixar_saeb(c.data_dir)); print(etl.verificar_checksums(c.data_dir))"

checksums:       ## confere data/raw contra data/raw/SHA256SUMS
	uv run python -c "from saeb_alfab import etl; from saeb_alfab.config import Config; print(etl.verificar_checksums(Config().data_dir))"

dados-snapshot:  ## restaura data/raw a partir do snapshot congelado (mesmos dados do TCC; não depende do Inep)
	cd data/snapshot && shasum -a 256 -c SHA256SUMS && tar -xJf saeb2023_2ef_snapshot.tar.xz -C ../raw && cd ../raw && shasum -a 256 -c SHA256SUMS

pdf:             ## compila a monografia com Tectonic (leve; brew install tectonic) → monografia/build/projeto.pdf
	cd monografia && tectonic -X compile projeto.tex --outdir build
	@echo "PDF: monografia/build/projeto.pdf"

-include Makefile.local
