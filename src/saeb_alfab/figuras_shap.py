"""Figuras da explicabilidade do modelo selecionado (Seção 3.8): resumo do SHAP, dependência e SHAP × LIME.

Feitas para impressão: paleta validada (azul e laranja), marcador ou hachura própria por série para leitura em
preto e branco, decimais com vírgula. Na trilha de proporção, o SHAP está na escala da margem do modelo
(log-odds) e o LIME na da proporção prevista. Por isso a comparação SHAP × LIME usa a participação com sinal de
cada variável no total de cada método, que é o que as medidas de concordância (Jaccard, Spearman, sinal) comparam.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AZUL, LARANJA, TINTA, TINTA2, GRADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequencial azul, claro → escuro
CASOS_TEXTO = ("típica (menor |resíduo|)", "acima do esperado (maior resíduo +)", "abaixo do esperado (maior resíduo −)")


def slug_caso(rotulo: str) -> str:
    """Nome de arquivo único por caso (‘extremo inferior’ e ‘extremo superior’ não podem colidir)."""
    return rotulo.split(" (")[0].replace(" do esperado", "").replace(" ", "_")


def _parte_especifica(desc: str) -> str:
    """O dicionário traz o enunciado do bloco e, no fim, o item: fica a parte que identifica o item."""
    txt = " ".join(str(desc).split())
    for sep in (": ", "? ", ". "):                   # do separador mais específico ao mais genérico
        if sep in txt and txt.rsplit(sep, 1)[1].strip():
            return txt.rsplit(sep, 1)[1].strip()
    return txt


def rotulos(meta_sel: pd.DataFrame, largura: int = 64) -> dict:
    """Rótulo = tema + parte específica do item + [P]/[D] (professor ou diretor)."""
    d = {}
    for _, r in meta_sel.iterrows():
        v = r.variavel
        if v == "ESC_INSE":
            d[v] = "INSE da escola"
            continue
        txt = f"{r.tema}: {_parte_especifica(r.descricao)}" if isinstance(r.tema, str) else _parte_especifica(r.descricao)
        if v.endswith("__FORA_ESCALA"):
            txt += " (fora da escala)"
        txt = (txt[:largura - 1] + "…") if len(txt) > largura else txt
        bloco = "P" if v.startswith("PROF_") else ("D" if v.startswith("DIR_") else "")
        d[v] = f"{txt} [{bloco}]" if bloco else txt
    return d


def _estilo(ax, eixo_y_grade=False):
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=7.5)
    (ax.yaxis if eixo_y_grade else ax.xaxis).grid(True, color=GRADE, lw=.6)
    ax.set_axisbelow(True)


def _cor_valor(x: np.ndarray) -> list:
    """Valor da variável na rampa sequencial (por posto, para não depender da escala). Ausente = cinza."""
    r = pd.Series(x).rank(pct=True).values
    return [("#c8c7c2" if np.isnan(v) else RAMPA[min(int(v * len(RAMPA)), len(RAMPA) - 1)]) for v in r]


def figura_resumo(sv: pd.DataFrame, valores: pd.DataFrame, rot: dict, titulo: str, k: int = 12):
    """Uma linha por variável (as k de maior |SHAP| médio), um ponto por escola do teste."""
    import matplotlib.pyplot as plt
    top = sv.abs().mean().sort_values(ascending=False).head(k).index[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * k + 1.2))
    rng = np.random.default_rng(0)
    for i, v in enumerate(top):
        x = sv[v].values
        cor = _cor_valor(valores[v].values.astype(float)) if v in valores else ["#c8c7c2"] * len(x)
        ax.scatter(x, i + rng.uniform(-.28, .28, len(x)), s=9, c=cor, edgecolor="none")
    ax.axvline(0, color=TINTA2, lw=.8)
    ax.set_yticks(range(len(top)), [rot.get(v, v) for v in top], fontsize=7.5)
    ax.set(xlabel="contribuição SHAP para a previsão da escola", title=titulo)
    _estilo(ax)
    ax.text(1.0, -0.13, "cor: valor da variável (claro = baixo, escuro = alto; cinza = ausente ou nominal)",
            transform=ax.transAxes, ha="right", fontsize=7, color=TINTA2)
    fig.tight_layout()
    return fig


def figura_dependencia(sv: pd.DataFrame, valores: pd.DataFrame, rot: dict, titulo: str, k: int = 4):
    import matplotlib.pyplot as plt
    top = [v for v in sv.abs().mean().sort_values(ascending=False).index if v in valores][:k]
    fig, axs = plt.subplots(1, len(top), figsize=(2.4 * len(top), 2.6), sharey=True)
    for ax, v in zip(np.atleast_1d(axs), top):
        ax.scatter(valores[v], sv[v], s=10, color=AZUL, edgecolor="white", linewidth=.4)
        ax.axhline(0, color=TINTA2, lw=.6)
        ax.set_title(rot.get(v, v), fontsize=7.5)
        ax.set_xlabel("valor da variável", fontsize=7.5)
        _estilo(ax, eixo_y_grade=True)
    np.atleast_1d(axs)[0].set_ylabel("contribuição SHAP", fontsize=7.5)
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    return fig


def _participacao(s: pd.Series) -> pd.Series:
    tot = s.abs().sum()
    return s / tot if tot > 0 else s


def figura_lime_shap(detalhes: dict, rot: dict, titulo: str, k: int = 8):
    """Para cada caso escolhido, as k variáveis de maior |participação| em qualquer dos métodos."""
    import matplotlib.pyplot as plt
    casos = [c for c in CASOS_TEXTO if c in detalhes]
    fig, axs = plt.subplots(len(casos), 1, figsize=(7.6, (0.30 * k + 0.9) * len(casos)))
    for ax, caso in zip(np.atleast_1d(axs), casos):
        df = detalhes[caso]
        p = pd.DataFrame({"SHAP": _participacao(df["SHAP"]), "LIME": _participacao(df["LIME_media"])})
        p = p.loc[p.abs().max(axis=1).sort_values(ascending=False).head(k).index[::-1]]
        y = np.arange(len(p))
        ax.barh(y + .19, 100 * p["SHAP"], height=.36, color=AZUL, label="SHAP")
        ax.barh(y - .19, 100 * p["LIME"], height=.36, color=LARANJA, hatch="////", edgecolor="white", linewidth=0,
                label="LIME")
        ax.axvline(0, color=TINTA2, lw=.8)
        ax.set_yticks(y, [rot.get(v, v) for v in p.index], fontsize=6.8)
        ax.set_title(caso.split(" (")[0].capitalize(), fontsize=8.5)
        ax.set_xlabel("participação com sinal (%)", fontsize=7.5)
        _estilo(ax)
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(titulo, fontsize=9.5)
    fig.tight_layout()
    return fig


def gerar(pipe, pop, meta_sel, idx_te, idx_tr, detalhes: dict, rotulo_trilha: str) -> dict:
    """Todas as figuras de uma trilha. ``detalhes`` vem de ``explicabilidade.lime_vs_shap``."""
    from .explicabilidade import shap_local
    sv = shap_local(pipe, pop, meta_sel, list(idx_te), idx_tr)
    numericas = meta_sel[meta_sel.tipo != "nominal"].variavel
    valores = pop.loc[idx_te, [v for v in numericas if v in sv.columns]].astype(float)
    rot = rotulos(meta_sel)
    figs = {"resumo": figura_resumo(sv, valores, rot, f"{rotulo_trilha}: contribuições SHAP no teste (12 variáveis principais)"),
            "lime_shap": figura_lime_shap(detalhes, rot, f"{rotulo_trilha}: SHAP × LIME nos casos do texto")}
    # Num modelo linear, a dependência de cada variável é uma reta (coeficiente × valor) e nada acrescenta ao
    # resumo. Os gráficos de dependência só são gerados quando o modelo selecionado é um ensemble de árvores.
    mdl = pipe.named_steps["model"]
    if hasattr(mdl, "estimators_") or hasattr(mdl, "booster_") or hasattr(mdl, "get_booster"):
        figs["dependencia"] = figura_dependencia(sv, valores, rot, f"{rotulo_trilha}: dependência das 4 variáveis principais")
    return figs
