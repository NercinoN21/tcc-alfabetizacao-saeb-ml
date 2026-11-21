"""Extração dos microdados do Saeb 2023 (mini ETL) e verificação de integridade.

O zip do Inep (~700 MB) é aberto remotamente por HTTP Range e só os arquivos necessários
são extraídos. Correção de segurança em relação ao notebook legado: o certificado
intermediário do Inep (que o servidor não envia) é VERSIONADO em `certs/` e conferido
por impressão digital SHA-256, em vez de ser baixado por HTTP simples e aceito às cegas.
Os CSVs são conferidos contra `data/raw/SHA256SUMS` (o Inep republica o zip).
"""
from __future__ import annotations

import hashlib
import io
import ssl
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

from .config import ARQUIVOS_SAEB, URL_INEP

CERT_INTERMEDIARIO = Path(__file__).parent / "certs" / "rnp_icpedu_gr46_ov_tls_ca_2025.pem"
SHA256_CERT = "E10747D4DA7BAB09CBA9952F019D3534CB9FBA070BF13D8791B1699CD2FF59DD"

CSV_KW = dict(sep=";", encoding="latin1", low_memory=False)
ALUNO_COLS = ["ID_REGIAO", "ID_UF", "ID_MUNICIPIO", "ID_AREA", "ID_ESCOLA", "IN_PUBLICA", "ID_LOCALIZACAO",
              "ID_TURMA", "IN_SITUACAO_CENSO", "IN_PREENCHIMENTO_LP", "IN_PROFICIENCIA_LP", "IN_AMOSTRA",
              "PESO_ALUNO_LP", "IN_ALFABETIZADO", "PROFICIENCIA_LP_SAEB"]
ESCOLA_COLS = ["ID_ESCOLA", "IN_PUBLICA", "NIVEL_SOCIO_ECONOMICO", "PC_FORMACAO_DOCENTE_INICIAL"]


# ------------------------------------------------------------------ integridade
def sha256(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def verificar_checksums(pasta: Path, arquivo_somas: str = "SHA256SUMS") -> dict[str, bool]:
    """Confere cada arquivo listado em SHA256SUMS. Levanta erro se algum divergir."""
    somas = {}
    for linha in (pasta / arquivo_somas).read_text().splitlines():
        if linha.strip():
            h, nome = linha.split(maxsplit=1)
            somas[nome.strip()] = h
    resultado = {nome: sha256(pasta / nome) == h for nome, h in somas.items() if (pasta / nome).exists()}
    divergentes = [n for n, ok in resultado.items() if not ok]
    if divergentes:
        raise ValueError(f"Checksum divergente (dados diferentes dos usados no TCC): {divergentes}")
    return resultado


# ------------------------------------------------------------------ TLS com certificado fixado
def contexto_verificacao() -> str:
    """Devolve um pacote de CAs = certifi + intermediário versionado (conferido por SHA-256)."""
    import certifi

    pem = CERT_INTERMEDIARIO.read_text()
    der = ssl.PEM_cert_to_DER_cert(pem)
    if hashlib.sha256(der).hexdigest().upper() != SHA256_CERT:
        raise ValueError("Certificado intermediário versionado foi alterado (impressão digital não confere).")
    pacote = Path(tempfile.gettempdir()) / "saeb_ca_bundle.pem"
    pacote.write_text(Path(certifi.where()).read_text() + "\n" + pem)
    return str(pacote)


class ArquivoHTTP(io.RawIOBase):
    """Arquivo remoto 'seekable' via HTTP Range, para o `zipfile` ler só o que precisa."""

    def __init__(self, url: str, verify):
        import requests

        self.url, self.pos, self.sessao = url, 0, requests.Session()
        self.sessao.verify = verify
        r = self.sessao.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("o servidor não aceita requisições Range")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, desloc, quando=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[quando]
        self.pos = base + desloc
        return self.pos

    def readinto(self, buf):
        import requests

        if self.pos >= self.size:
            return 0
        fim = min(self.pos + len(buf), self.size) - 1
        for tent in range(4):
            try:
                r = self.sessao.get(self.url, headers={"Range": f"bytes={self.pos}-{fim}"}, timeout=180)
                if r.status_code == 206:
                    n = len(r.content)
                    buf[:n] = r.content
                    self.pos += n
                    return n
            except requests.RequestException:
                pass
            time.sleep(2 * (tent + 1))
        raise IOError("falha ao ler o intervalo de bytes no servidor do Inep")


def _extrair(zf: zipfile.ZipFile, nomes, destino: Path):
    por_nome = {Path(i.filename).name.lower(): i for i in zf.infolist() if not i.is_dir()}
    for nome in nomes:
        info = por_nome.get(nome.lower())
        if info is None:
            raise FileNotFoundError(f"{nome} não está no zip do Inep")
        tmp = destino / (nome + ".part")
        with zf.open(info) as src, open(tmp, "wb") as dst:
            for bloco in iter(lambda: src.read(1 << 20), b""):
                dst.write(bloco)
        tmp.replace(destino / nome)
        print(f"  ✓ {nome:28s} {info.file_size / 1e6:8.1f} MB")


def baixar_saeb(destino: Path, url: str = URL_INEP, necessarios=ARQUIVOS_SAEB) -> str:
    """Garante os arquivos necessários em `destino`. Devolve a origem usada."""
    destino.mkdir(parents=True, exist_ok=True)
    faltam = [f for f in necessarios if not (destino / f).exists()]
    if not faltam:
        return "cache local"
    print("Baixando do Inep:", ", ".join(faltam))
    with zipfile.ZipFile(io.BufferedReader(ArquivoHTTP(url, contexto_verificacao()), buffer_size=8 << 20)) as zf:
        _extrair(zf, faltam, destino)
    return "download parcial (HTTP Range)"


# ------------------------------------------------------------------ leitura
def ler_bases(pasta: Path):
    """Lê as quatro bases e o dicionário. Devolve (aluno, escola, prof, dire, planilhas)."""
    aluno = pd.read_csv(pasta / "TS_ALUNO_2EF.csv", usecols=ALUNO_COLS, **CSV_KW)
    escola = pd.read_csv(pasta / "TS_ESCOLA.csv", usecols=ESCOLA_COLS, **CSV_KW)
    prof = pd.read_csv(pasta / "TS_PROFESSOR.csv", **CSV_KW)
    dire = pd.read_csv(pasta / "TS_DIRETOR.csv", **CSV_KW)
    planilhas = pd.read_excel(pasta / "Dicionario_Saeb_2023.xlsx", sheet_name=None, header=None)
    return aluno, escola, prof, dire, planilhas
