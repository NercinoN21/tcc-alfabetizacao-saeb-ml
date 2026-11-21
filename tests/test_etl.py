"""Integridade dos dados e do certificado fixado."""
import pytest

from saeb_alfab import etl


def test_checksum_divergente_e_detectado(tmp_path):
    (tmp_path / "a.csv").write_text("x;y\n1;2\n")
    (tmp_path / "SHA256SUMS").write_text(f"{etl.sha256(tmp_path / 'a.csv')}  a.csv\n")
    assert etl.verificar_checksums(tmp_path) == {"a.csv": True}
    (tmp_path / "a.csv").write_text("x;y\n1;3\n")
    with pytest.raises(ValueError):
        etl.verificar_checksums(tmp_path)


def test_certificado_versionado_confere_com_impressao_digital():
    assert etl.contexto_verificacao().endswith(".pem")
