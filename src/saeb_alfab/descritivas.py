"""Análises descritivas da base por escola (capítulo de Resultados), portadas do notebook do TCC 1.

Toda média que descreve a rede sai em duas versões: sem peso (a amostra analisada) e com o peso amostral
da escola (`PESO`, decisão D2 e Seção 3.3). As tabelas cobrem:
* números da unidade de análise (alunos, turmas, escolas, ICC) para a coorte interna e a população principal;
* desfechos, geografia e INSE (o INSE sozinho e quanto da variação entre escolas ele descreve);
* volatilidade da proporção em escolas pequenas (funil binomial);
* cobertura e ausência por variável e por tema, e se a ausência de questionário é informativa;
* viés de seleção associado à disponibilidade do INSE (diferença padronizada de médias, SMD);
* composição de cor/raça de professores e diretores e sua associação bruta e por faixa de INSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import AREA, LOCALIZACAO, REGIAO
from .robustez import grupo_inse

DESFECHOS = {"Y_PROP_ALFAB": "proporção de alfabetizados", "Y_PROF_MEDIA": "proficiência média em LP"}


# ------------------------------------------------------------------ estatísticas básicas
def icc1(df: pd.DataFrame, grupo: str, y: str) -> float:
    """Correlação intraclasse por ANOVA de um fator: fração da variância individual situada entre grupos."""
    g = df.groupby(grupo)[y]
    n_i, k, n = g.size(), g.ngroups, len(df)
    media_g = g.mean()
    msb = float((n_i * (media_g - df[y].mean()) ** 2).sum() / (k - 1))
    msw = float(((df[y] - df[grupo].map(media_g)) ** 2).sum() / (n - k))
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def smd(a, b) -> float:
    """Diferença padronizada de médias, com o desvio-padrão combinado sqrt((var_a + var_b) / 2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dp = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / dp) if dp > 0 else 0.0


def _media_pond(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


# ------------------------------------------------------------------ unidade de análise (Seção 3.2.1)
def numeros_unidade(alunos: pd.DataFrame, nome: str) -> dict:
    return {"coorte": nome, "alunos": len(alunos), "turmas": alunos.ID_TURMA.nunique(),
            "escolas": alunos.ID_ESCOLA.nunique(),
            "ICC proficiência (escola)": icc1(alunos, "ID_ESCOLA", "PROFICIENCIA_LP_SAEB"),
            "ICC proficiência (turma)": icc1(alunos, "ID_TURMA", "PROFICIENCIA_LP_SAEB"),
            "ICC alfabetizado (escola)": icc1(alunos, "ID_ESCOLA", "ALFAB"),
            "% escolas com mais de uma turma": 100 * float((alunos.groupby("ID_ESCOLA").ID_TURMA.nunique() > 1).mean())}


def tabela_unidade(alunos_interna: pd.DataFrame, alunos_validos: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    pop_alunos = alunos_validos[alunos_validos.ID_ESCOLA.isin(pop.index)]
    return pd.DataFrame([numeros_unidade(alunos_interna, "coorte interna (professor e diretor)"),
                         numeros_unidade(pop_alunos, "população principal (junção à esquerda)")]).set_index("coorte").T


# ------------------------------------------------------------------ desfechos, geografia e INSE
def estatisticas_desfechos(pop: pd.DataFrame) -> pd.DataFrame:
    cols = ["N_ALUNOS", "CTX_N_TURMAS", "Y_PROP_ALFAB", "Y_PROF_MEDIA", "Y_PROF_MEDIANA"]
    t = pop[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
    t["média com peso"] = [_media_pond(pop[c], pop.PESO) for c in cols]
    t["corr. com a mediana"] = [np.nan, np.nan, np.nan, float(pop.Y_PROF_MEDIA.corr(pop.Y_PROF_MEDIANA)), np.nan]
    return t


def _resumo_grupo(g: pd.DataFrame) -> pd.Series:
    d = {"escolas": len(g), "alunos": int(g.N_ALUNOS.sum()), "INSE médio": g.ESC_INSE.mean()}
    for y, rot in DESFECHOS.items():
        d[f"{rot} (sem peso)"] = g[y].mean()
        d[f"{rot} (com peso)"] = _media_pond(g[y], g.PESO)
    return pd.Series(d)


def por_geografia(pop: pd.DataFrame) -> pd.DataFrame:
    rot = {"Região": pop.GEO_ID_REGIAO.map(REGIAO), "Localização": pop.GEO_ID_LOCALIZACAO.map(LOCALIZACAO),
           "Área": pop.GEO_ID_AREA.map(AREA)}
    partes = {eixo: pop.groupby(s).apply(_resumo_grupo, include_groups=False) for eixo, s in rot.items()}
    return pd.concat(partes, names=["eixo", "grupo"])


def por_inse(pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Desfechos por nível de INSE e o quanto o INSE sozinho descreve (Spearman e R² da média por nível)."""
    nivel = pop.ESC_INSE.rename("nível de INSE")
    t = pop.groupby(nivel).apply(_resumo_grupo, include_groups=False).drop(columns="INSE médio")
    linhas = []
    for y, rot in DESFECHOS.items():
        media_nivel = pop.groupby("ESC_INSE")[y].transform("mean")
        r2 = 1 - float(((pop[y] - media_nivel) ** 2).sum() / ((pop[y] - pop[y].mean()) ** 2).sum())
        linhas.append({"desfecho": rot, "Spearman (INSE, desfecho)": float(stats.spearmanr(pop.ESC_INSE, pop[y]).statistic),
                       "R² da média por nível de INSE": r2})
    return t, pd.DataFrame(linhas).set_index("desfecho")


def funil_binomial(pop: pd.DataFrame) -> pd.DataFrame:
    """Fração de escolas fora da banda de ±1,96 desvio-padrão binomial em torno da proporção geral: se a
    variação entre escolas fosse só acaso amostral, cerca de 5% ficariam fora."""
    p0 = float(np.average(pop.Y_PROP_ALFAB, weights=pop.N_ALUNOS))
    banda = 1.96 * np.sqrt(p0 * (1 - p0) / pop.N_ALUNOS)
    fora = (pop.Y_PROP_ALFAB - p0).abs() > banda
    return pd.Series({"proporção geral (ponderada por alunos)": p0, "escolas": len(pop),
                      "% fora da banda de 95% do acaso": 100 * float(fora.mean())}, name="valor").to_frame()


# ------------------------------------------------------------------ cobertura e ausência (Seções 3.1.2 e 3.5.2)
def cobertura_por_variavel(pop, decisoes, prof_cod_turma, dir_cod_escola) -> pd.DataFrame:
    """Por item: não resposta entre quem recebeu o questionário, ausência na base por escola (a medida da regra
    dos 30%) e a mesma ausência ponderada pelo peso amostral."""
    pc = prof_cod_turma[prof_cod_turma.ID_ESCOLA.isin(pop.index)]
    dc = dir_cod_escola[dir_cod_escola.index.isin(pop.index)]
    bruta = pd.concat([pc[[c for c in pc.columns if c.startswith("PROF_")]].isna().mean(), dc.isna().mean()])
    t = decisoes.set_index("variavel")[["tema", "tipo", "decisao", "ausencia"]].rename(
        columns={"ausencia": "ausência na base por escola"})
    t["não resposta entre quem recebeu"] = bruta.reindex(t.index)
    t["ausência ponderada pelo peso"] = [_media_pond(pop[v].isna(), pop.PESO) if v in pop else np.nan for v in t.index]
    return t


def cobertura_por_tema(cob: pd.DataFrame) -> pd.DataFrame:
    return cob.groupby("tema").agg(
        variaveis=("tipo", "size"), ausencia_mediana=("ausência na base por escola", "median"),
        nao_resposta_mediana=("não resposta entre quem recebeu", "median"),
        incluidas=("decisao", lambda s: int((s == "incluída").sum())),
        cortadas_por_ausencia=("decisao", lambda s: int(s.str.contains("ausência").sum())),
    ).sort_values("ausencia_mediana", ascending=False)


def ausencia_informativa(pop: pd.DataFrame) -> pd.DataFrame:
    """Escolas sem questionário de professor ou de diretor diferem das demais nos desfechos?"""
    linhas = []
    for bloco, col in (("professor", "TEM_PROF"), ("diretor", "TEM_DIR")):
        for y, rot in DESFECHOS.items():
            com, sem = pop.loc[pop[col], y], pop.loc[~pop[col], y]
            linhas.append({"questionário": bloco, "desfecho": rot, "escolas sem": len(sem),
                           "média com": com.mean(), "média sem": sem.mean(), "SMD (com − sem)": smd(com, sem),
                           "p (Mann-Whitney)": stats.mannwhitneyu(com, sem).pvalue if len(sem) > 2 else np.nan})
    return pd.DataFrame(linhas).set_index(["questionário", "desfecho"])


def vies_inse(base_todas: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """Dentro da rede pública com n mínimo, escolas com e sem INSE (Seção 3.4). |SMD| > 0,2 sugere desequilíbrio."""
    pub = base_todas[(base_todas.PUBLICA == 1) & (base_todas.N_ALUNOS >= n_min)].copy()
    pub["rural"] = (pub.GEO_ID_LOCALIZACAO == "2").astype(float)
    pub["capital"] = (pub.GEO_ID_AREA == "1").astype(float)
    pub["tem professor"], pub["tem diretor"] = pub.TEM_PROF.astype(float), pub.TEM_DIR.astype(float)
    for k, nome in REGIAO.items():
        pub[f"região {nome}"] = (pub.GEO_ID_REGIAO == k).astype(float)
    variaveis = ["Y_PROP_ALFAB", "Y_PROF_MEDIA", "N_ALUNOS", "CTX_N_TURMAS", "rural", "capital", "tem professor",
                 "tem diretor"] + [f"região {n}" for n in REGIAO.values()]
    a, b = pub[pub.TEM_INSE], pub[~pub.TEM_INSE]
    linhas = [{"variável": v, f"com INSE (n = {len(a)})": a[v].mean(), f"sem INSE (n = {len(b)})": b[v].mean(),
               "SMD": smd(a[v], b[v]), "p (Mann-Whitney)": stats.mannwhitneyu(a[v].dropna(), b[v].dropna()).pvalue}
              for v in variaveis]
    return pd.DataFrame(linhas).set_index("variável")


# ------------------------------------------------------------------ cor/raça (Seção 3.5.3, auditoria)
def _grupos_cor(pop: pd.DataFrame) -> pd.DataFrame:
    c = pop.copy()
    c["faixa de INSE"] = c.ESC_INSE.map(grupo_inse)
    c["professores"] = np.select([c.PROF_COR_PCT_PRETA_PARDA.isna(), c.PROF_COR_PCT_PRETA_PARDA > 0.5],
                                 ["sem questionário", "maioria preta ou parda"], "minoria preta ou parda")
    c["diretor preto ou pardo"] = c.DIR_COR.map({"Preta": 1.0, "Parda": 1.0, "Branca": 0.0})
    return c


def composicao_cor(pop: pd.DataFrame) -> pd.DataFrame:
    c = _grupos_cor(pop)
    f = lambda g: pd.Series({"escolas": len(g), "% de turmas com professor preto ou pardo": 100 * g.PROF_COR_PCT_PRETA_PARDA.mean(),
                             "% de diretores pretos ou pardos (entre brancos, pretos e pardos)": 100 * g["diretor preto ou pardo"].mean()})
    eixos = {"Região": c.GEO_ID_REGIAO.map(REGIAO), "Faixa de INSE": c["faixa de INSE"],
             "Localização": c.GEO_ID_LOCALIZACAO.map(LOCALIZACAO)}
    return pd.concat({k: c.groupby(s).apply(f, include_groups=False) for k, s in eixos.items()}, names=["eixo", "grupo"])


def associacao_cor(pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associação bruta entre cor/raça e desfechos e a mesma associação dentro de cada faixa de INSE. Se a
    diferença encolhe dentro das faixas, a associação bruta é sobretudo composição."""
    c = _grupos_cor(pop)
    agg = dict(escolas=("N_ALUNOS", "size"), prop_alfab=("Y_PROP_ALFAB", "mean"), prof_media=("Y_PROF_MEDIA", "mean"),
               inse_medio=("ESC_INSE", "mean"))
    bruta = pd.concat({"cor do diretor": c.groupby("DIR_COR").agg(**agg),
                       "professores": c.groupby("professores").agg(**agg)}, names=["variável", "grupo"])
    d = c[c.professores != "sem questionário"]
    estrato = d.pivot_table(index="faixa de INSE", columns="professores", values="Y_PROP_ALFAB", aggfunc=["mean", "size"])
    return bruta, estrato
