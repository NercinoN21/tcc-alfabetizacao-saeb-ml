"""Dados do simulador do painel: escolas do mesmo nível socioeconômico, previsão do modelo e resultado real.

Por faixa de INSE, compara a previsão fora da amostra do modelo selecionado (média do grupo) com a distribuição
real da proporção de alfabetizados nas escolas do grupo, resumida em percentis (10, 25, 50, 75 e 90). Só
estatísticas agregadas saem daqui. Nenhuma escola individual é exportada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAIXAS = {2: "I–II", 3: "III", 4: "IV", 5: "V", 6: "VI–VII"}


def distribuicao_por_inse(pop: pd.DataFrame, alvo: str, previsto_oof: np.ndarray) -> pd.DataFrame:
    faixa = pop["ESC_INSE"].round().clip(2, 6).astype(int)
    t = pd.DataFrame({"faixa": faixa, "obs": pop[alvo].values, "prev": previsto_oof})
    g = t.groupby("faixa")
    out = pd.DataFrame({"escolas": g.size(), "previsto_medio": g.prev.mean(),
                        "erro_absoluto_medio": g.apply(lambda d: (d.obs - d.prev).abs().mean(), include_groups=False)})
    for q in (10, 25, 50, 75, 90):
        out[f"observado_p{q}"] = g.obs.quantile(q / 100)
    out.index = out.index.map(FAIXAS)
    return out
