"""Protótipo do painel: só agregados, com supressão de células pequenas (LGPD, art. 12).

Duas partes:
- ``agregados``: tabelas por UF, região, localização e faixa de INSE, com desfechos sem e com peso amostral e o
  resíduo médio do modelo selecionado (observado − previsto fora da amostra). Grupos com menos de ``n_min`` escolas
  têm todos os valores suprimidos. Nenhuma escola individual aparece.
- ``gerar_html``: página autônoma (modelo em ``painel_modelo.html``, sem bibliotecas externas) com os dados de
  ``dados_pagina``, lidos das tabelas congeladas em reports/final.

Uso: python -m saeb_alfab.painel <pasta_painel> <reports/final> <saida.html>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOCALIZACAO, REGIAO, UF

DESFECHOS = {"Y_PROP_ALFAB": "prop", "Y_PROF_MEDIA": "prof"}
FAIXAS_INSE = {1: "I–II", 2: "I–II", 3: "III", 4: "IV", 5: "V", 6: "VI–VII", 7: "VI–VII", 8: "VI–VII"}


def _media_pond(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def _resumo(g: pd.DataFrame, n_boot: int = 2000, seed: int = 2110) -> pd.Series:
    d = {"escolas": len(g), "alunos": int(g.N_ALUNOS.sum()), "INSE médio": g.ESC_INSE.mean()}
    rng = np.random.default_rng(seed)
    for y, k in DESFECHOS.items():
        d[f"{k} sem peso"] = g[y].mean()
        d[f"{k} com peso"] = _media_pond(g[y], g.PESO)
        d[f"{k} com peso IC inf"], d[f"{k} com peso IC sup"] = _ic_pond(g, y, n_boot, rng)
        d[f"{k} resíduo médio"] = g[f"_res_{k}"].mean()
        d[f"{k} erro absoluto médio"] = g[f"_res_{k}"].abs().mean()
    return pd.Series(d)


def _ic_pond(g: pd.DataFrame, y: str, n_boot: int, rng) -> tuple[float, float]:
    """IC95% da média ponderada pelo peso amostral, por bootstrap das escolas do grupo."""
    v, w = g[y].to_numpy(float), g.PESO.to_numpy(float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    est = (v[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return float(np.quantile(est, .025)), float(np.quantile(est, .975))


def suprimir(t: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """Grupos com menos de n_min escolas: todos os valores viram NaN e a coluna 'suprimido' vale True."""
    t = t.copy()
    pequeno = t["escolas"] < n_min
    t.loc[pequeno, t.select_dtypes("number").columns] = np.nan
    t["suprimido"] = pequeno
    return t


def agregados(pop: pd.DataFrame, previsto: dict, n_min: int) -> dict:
    """``previsto``: {'prop': Series, 'prof': Series} com a previsão fora da amostra, no índice de ``pop``."""
    p = pop.copy()
    for y, k in DESFECHOS.items():
        p[f"_res_{k}"] = p[y] - previsto[k].reindex(p.index)
    eixos = {"uf": p.GEO_ID_UF.astype(int).astype(str).map(UF),
             "regiao": p.GEO_ID_REGIAO.astype(int).astype(str).map(REGIAO),
             "localizacao": p.GEO_ID_LOCALIZACAO.astype(int).astype(str).map(LOCALIZACAO),
             "inse": p.ESC_INSE.round().astype(int).map(FAIXAS_INSE)}
    out = {}
    for nome, s in eixos.items():
        t = p.groupby(s.rename("grupo")).apply(_resumo, include_groups=False)
        if nome == "uf":
            reg = p.groupby(s.rename("grupo")).GEO_ID_REGIAO.first().astype(int).astype(str).map(REGIAO)
            t.insert(0, "região", reg)
        out[nome] = suprimir(t, n_min)
    out["total"] = suprimir(pd.DataFrame([_resumo(p)], index=pd.Index(["Brasil (base analítica)"], name="grupo")), n_min)
    return out


# ---------------------------------------------------------------------------------------------------- página
def _registros(t: pd.DataFrame) -> list:
    t = t.reset_index()
    return json.loads(t.to_json(orient="records", force_ascii=False))


def dados_pagina(pasta_painel: Path, pasta_final: Path, n_min: int, versao: str) -> dict:
    """Tudo o que a página mostra, lido das tabelas congeladas (nenhum número digitado)."""
    d = {k: _registros(pd.read_csv(pasta_painel / f"05_painel_{k}.csv", index_col=0))
         for k in ("uf", "regiao", "localizacao", "inse", "total")}
    ordem = ["I–II", "III", "IV", "V", "VI–VII"]
    d["inse"].sort(key=lambda r: ordem.index(r["grupo"]) if r["grupo"] in ordem else 99)
    for k in ("uf", "regiao", "localizacao", "inse", "total"):
        for r in d[k]:
            r["suprimido"] = bool(r.get("suprimido"))
    rede = pd.read_csv(pasta_final / "complemento_pesos/tabelas/01_alfabetizacao_por_rede.csv", index_col=0)
    d["rede"] = {k: {"com": float(rede.loc[k, "proporção com peso (%)"]), "sem": float(rede.loc[k, "proporção sem peso (%)"]),
                     "alunos": int(rede.loc[k, "alunos"])} for k in rede.index}
    ic = pd.read_csv(pasta_final / "complemento_pesos/tabelas/01_alfabetizacao_por_rede_ic.csv", index_col=0)
    for k in d["rede"]:
        d["rede"][k]["ic"] = [float(ic.loc[k, "IC95% inf (%)"]), float(ic.loc[k, "IC95% sup (%)"])]
    so = pd.read_csv(pasta_final / "complemento_descritivas/tabelas/01_inse_sozinho.csv", index_col=0)
    d["inse_r2"] = {"prop": float(so.iloc[0, 1]), "prof": float(so.iloc[1, 1])}
    abl, temas = {}, {}
    for tr in ("prop", "prof"):
        a = pd.read_csv(pasta_final / f"complemento_ablacao_indices/tabelas/04_ablacao_{tr}.csv", index_col=0)
        m1, m4 = a.loc["M1: INSE + geografia"], a.loc["M4: completo"]
        rel = lambda k: float(100 * a.loc[k, "Δ MAE vs M4"] / m4["MAE (CV)"])
        abl[tr] = {"ganho": {k: float(a.loc[k, "ganho vs baseline (%)"]) for k in
                             ("M0: só INSE", "M1: INSE + geografia", "M4: completo")},
                   "mae_m1": float(m1["MAE (CV)"]), "delta": float(m4["Δ MAE vs M1"]), "ic": [float(m4["IC95% inf vs M1"]), float(m4["IC95% sup vs M1"])],
                   "reducao_pct": float(-100 * m4["Δ MAE vs M1"] / m1["MAE (CV)"]),
                   "retirada": {"professor": rel("M4 − professor"), "diretor": rel("M4 − diretor"),
                                "INSE": rel("M5: completo sem INSE"), "geografia": rel("M4 − geografia")},
                   "ic_retirada": {n: [float(a.loc[k, "IC95% inf"]), float(a.loc[k, "IC95% sup"])] for n, k in
                                   (("professor", "M4 − professor"), ("diretor", "M4 − diretor"),
                                    ("INSE", "M5: completo sem INSE"), ("geografia", "M4 − geografia"))}}
        e = pd.read_csv(pasta_final / f"tabelas/03_estabilidade_temas_{tr}.csv", index_col=0).head(6)
        temas[tr] = [{"tema": i.replace("—", "de").replace("Infraestrutura de Educação", "Infraestrutura da Educação"),
                      "participacao": float(r["participação média"]), "top3": float(r["% repetições no top-3"]),
                      "posicao": float(r["posição média"])} for i, r in e.iterrows()]
    d["ablacao"], d["temas"] = abl, temas
    f = pasta_painel / "05_simulador_distribuicao.csv"
    if f.exists():
        d["simulador"] = _registros(pd.read_csv(f, index_col=0))

    d["meta"] = {"n_min": n_min, "versao": versao,
                 "repo": "https://github.com/NercinoN21/tcc-alfabetizacao-saeb-ml"}
    return d


def gerar_html(pasta_painel: Path, pasta_final: Path, n_min: int, versao: str) -> str:
    modelo = (Path(__file__).parent / "painel_modelo.html").read_text()
    return modelo.replace("/*DADOS*/", json.dumps(dados_pagina(pasta_painel, pasta_final, n_min, versao), ensure_ascii=False))


if __name__ == "__main__":
    from .config import PAINEL_N_MIN
    pasta_painel, pasta_final, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    info = json.loads((pasta_painel.parent / "run_info.json").read_text()) if (pasta_painel.parent / "run_info.json").exists() else {}
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(gerar_html(pasta_painel, pasta_final, PAINEL_N_MIN, info.get("git", "?")))
    print(f"[ok] {saida}")
