"""Construção da base analítica no nível da escola (Seção 3.2) e do funil de cobertura.

Regras (inalteradas em relação ao TCC 1, exceto a codificação corrigida em `dicionario.py`):
* desfechos aluno -> escola: Y_PROP_ALFAB = média(1[prof >= 743]); Y_PROF_MEDIA; Y_PROF_MEDIANA;
* professor turma -> escola: numéricas/ordinais pela média; binárias pela proporção de turmas;
  nominais pela moda; nº de turmas retido (CTX_N_TURMAS);
* diretor, INSE (TS_ESCOLA) e geografia já são de nível escola;
* junção à esquerda: a escola só sai se perder TODOS os blocos de preditores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CORTE_ALFAB
from .dicionario import codificar_bloco

MAPA_INSE = {f"Nível {r}": i for i, r in enumerate(["I", "II", "III", "IV", "V", "VI", "VII", "VIII"], start=1)}
COR_DIR = {"A": "Branca", "B": "Preta", "C": "Parda", "D": "Amarela/Indígena", "E": "Amarela/Indígena",
           "F": "Não declarou"}


def moda(s: pd.Series):
    s = s.dropna()
    return s.value_counts().index[0] if len(s) else np.nan


def moda_int(s: pd.Series):
    m = moda(s)
    return str(int(m)) if pd.notna(m) else np.nan


def filtrar_questionarios(prof: pd.DataFrame, dire: pd.DataFrame):
    """Professor: 2º ano + bloco 'professor único' (SQ_QUESTIONARIO == 5) + preenchido. Diretor: 2º ano + preenchido."""
    p = prof[(prof.ID_SERIE == 2) & (prof.SQ_QUESTIONARIO == 5) & (prof.IN_PREENCHIMENTO_QUESTIONARIO == 1)].copy()
    d = dire[(dire.ID_SERIE == 2) & (dire.IN_PREENCHIMENTO_QUESTIONARIO == 1)].copy()
    return p, d


def unicidade_questionarios(prof: pd.DataFrame, dire: pd.DataFrame) -> pd.DataFrame:
    """Depois dos filtros, cada turma tem no máximo um questionário de professor e cada escola um de diretor
    (Seção 3.1). Reporta registros, chaves distintas e chaves repetidas."""
    p, d = filtrar_questionarios(prof, dire)
    linhas = {}
    for nome, df, chave in (("professor (por turma)", p, "ID_TURMA"), ("diretor (por escola)", d, "ID_ESCOLA")):
        c = df.groupby(chave).size()
        linhas[nome] = {"registros": len(df), "chaves distintas": int(c.size), "chaves com mais de um registro": int((c > 1).sum())}
    return pd.DataFrame(linhas).T


def aluno_valido(aluno: pd.DataFrame) -> pd.DataFrame:
    """Coorte válida (Seção 3.1.3). `IN_ALFABETIZADO` vale 0 também para quem não tem proficiência,
    por isso o filtro de proficiência calculável é obrigatório."""
    return aluno[(aluno.IN_SITUACAO_CENSO == 1) & (aluno.IN_PREENCHIMENTO_LP == 1)
                 & (aluno.IN_PROFICIENCIA_LP == 1) & aluno.PROFICIENCIA_LP_SAEB.notna()].copy()


def agregar_professor(prof_cod: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """turma -> escola: média (numérica/ordinal/binária -> proporção) e moda (nominal)."""
    tipos = meta.set_index("variavel")["tipo"]
    cols = [c for c in prof_cod.columns if c.startswith("PROF_")]
    num = [c for c in cols if tipos.get(c) != "nominal"]
    nom = [c for c in cols if tipos.get(c) == "nominal"]
    g = prof_cod.groupby("ID_ESCOLA")
    out = g[num].mean()
    for c in nom:
        out[c] = g[c].agg(moda)
    out["CTX_N_TURMAS"] = g["ID_TURMA"].nunique()
    return out


def derivar_cor_professor(prof_raw: pd.DataFrame) -> pd.DataFrame:
    d = prof_raw[["ID_ESCOLA", "TX_Q003"]].dropna(subset=["TX_Q003"]).copy()
    d["pp"], d["br"], d["ou"] = d.TX_Q003.isin(["B", "C"]), d.TX_Q003.eq("A"), d.TX_Q003.isin(["D", "E", "F"])
    g = d.groupby("ID_ESCOLA")
    return pd.DataFrame({"PROF_COR_PCT_PRETA_PARDA": g["pp"].mean(), "PROF_COR_PCT_BRANCA": g["br"].mean(),
                         "PROF_COR_PCT_OUTRA_ND": g["ou"].mean()})


def derivar_cor_diretor(dire_raw: pd.DataFrame) -> pd.DataFrame:
    s = dire_raw.drop_duplicates("ID_ESCOLA").set_index("ID_ESCOLA")["TX_Q009"].map(COR_DIR)
    return s.rename("DIR_COR").to_frame()


def construir_base(aluno, escola, prof, dire, meta, juncao: str = "esquerda", n_min: int = 1):
    """Base analítica por escola (todas as redes). Devolve (base, alunos_validos, prof_cod_turma, dir_cod_escola)."""
    a = aluno_valido(aluno)
    prof_f, dire_f = filtrar_questionarios(prof, dire)
    if juncao == "interna":
        a = a[a.ID_TURMA.isin(prof_f.ID_TURMA) & a.ID_ESCOLA.isin(dire_f.ID_ESCOLA)]
    a["ALFAB"] = (a.PROFICIENCIA_LP_SAEB >= CORTE_ALFAB).astype(int)
    if not (a.ALFAB == a.IN_ALFABETIZADO).all():
        raise AssertionError("IN_ALFABETIZADO diverge do corte de 743 do Inep")
    g = a.groupby("ID_ESCOLA")
    desf = pd.DataFrame({
        "N_ALUNOS": g.size(), "N_TURMAS_ALUNOS": g.ID_TURMA.nunique(),
        "Y_PROP_ALFAB": g.ALFAB.mean(), "Y_PROF_MEDIA": g.PROFICIENCIA_LP_SAEB.mean(),
        "Y_PROF_MEDIANA": g.PROFICIENCIA_LP_SAEB.median(),
        "PESO_ALUNOS_SOMA": g.PESO_ALUNO_LP.sum(), "PESO_ALUNO_MEDIO": g.PESO_ALUNO_LP.mean(),
        "PUBLICA": g.IN_PUBLICA.first(), "ID_MUNICIPIO": g.ID_MUNICIPIO.agg(moda_int),
        "GEO_ID_REGIAO": g.ID_REGIAO.agg(moda_int), "GEO_ID_UF": g.ID_UF.agg(moda_int),
        "GEO_ID_AREA": g.ID_AREA.agg(moda_int), "GEO_ID_LOCALIZACAO": g.ID_LOCALIZACAO.agg(moda_int),
    })
    prof_v = prof_f[prof_f.ID_TURMA.isin(a.ID_TURMA)].copy()
    prof_cod = codificar_bloco(prof_v, meta, "PROF")
    prof_cod["ID_ESCOLA"], prof_cod["ID_TURMA"] = prof_v["ID_ESCOLA"].values, prof_v["ID_TURMA"].values
    prof_agg = agregar_professor(prof_cod, meta)
    dire_v = dire_f[dire_f.ID_ESCOLA.isin(a.ID_ESCOLA)].drop_duplicates("ID_ESCOLA").copy()
    dir_cod = codificar_bloco(dire_v, meta, "DIR")
    dir_cod.index = dire_v["ID_ESCOLA"].values
    esc = escola.drop_duplicates("ID_ESCOLA").set_index("ID_ESCOLA")
    esc_f = pd.DataFrame({"ESC_INSE": esc["NIVEL_SOCIO_ECONOMICO"].map(MAPA_INSE),
                          "ESC_PC_FORMACAO_DOCENTE_INICIAL": esc["PC_FORMACAO_DOCENTE_INICIAL"]})
    base = (desf.join(prof_agg).join(derivar_cor_professor(prof_v)).join(dir_cod)
            .join(derivar_cor_diretor(dire_v)).join(dire_v.set_index("ID_ESCOLA")["VL_PESO_ESCOLA"].rename("PESO_ESCOLA"))
            .join(esc_f))
    base["TEM_PROF"], base["TEM_DIR"] = base.index.isin(prof_agg.index), base.index.isin(dir_cod.index)
    base["TEM_INSE"] = base["ESC_INSE"].notna()
    base["CTX_N_TURMAS"] = base["CTX_N_TURMAS"].fillna(0)
    # Ausência informativa: indicadores de escola sem o questionário (a mediana imputada não a distingue)
    base["PROF_SEM_QUESTIONARIO"] = (~base["TEM_PROF"]).astype(float)
    base["DIR_SEM_QUESTIONARIO"] = (~base["TEM_DIR"]).astype(float)
    base = base[(base.TEM_PROF | base.TEM_DIR) & (base.N_ALUNOS >= n_min)]
    return base, a, prof_cod, dir_cod


def peso_final(base: pd.DataFrame) -> pd.Series:
    """Peso amostral da escola (VL_PESO_ESCOLA, do TS_DIRETOR). Sem diretor, imputa
    PESO_ALUNO_MEDIO × razão mediana entre os dois pesos. Normalizado para média 1."""
    w = base["PESO_ESCOLA"].copy()
    razao = (base["PESO_ESCOLA"] / base["PESO_ALUNO_MEDIO"]).median()
    w = w.fillna(base["PESO_ALUNO_MEDIO"] * razao)
    return w / w.mean()


def populacoes(base_todas: pd.DataFrame, n_min: int):
    """(pop, universo): pop = rede pública com INSE e n mínimo (escopo principal);
    universo = público + privado com n mínimo (robustez, Seção 3.9)."""
    pop = base_todas[(base_todas.PUBLICA == 1) & base_todas.TEM_INSE & (base_todas.N_ALUNOS >= n_min)].copy()
    pop["PESO"] = peso_final(pop)
    universo = base_todas[base_todas.N_ALUNOS >= n_min].copy()
    universo["PESO"] = peso_final(universo)
    return pop, universo


def funil(aluno, escola, prof, dire, pop, alunos_interna, n_min: int) -> pd.DataFrame:
    """Tabela 3 do texto: alunos, turmas e escolas que restam em cada etapa."""
    def cont(nome, df):
        return dict(etapa=nome, alunos=len(df), turmas=df.ID_TURMA.nunique(), escolas=df.ID_ESCOLA.nunique())

    p, d = filtrar_questionarios(prof, dire)
    com_inse = set(escola.loc[escola.NIVEL_SOCIO_ECONOMICO.notna(), "ID_ESCOLA"])
    e = [cont("1. Registros iniciais (TS_ALUNO_2EF)", aluno)]
    a = aluno[aluno.IN_SITUACAO_CENSO == 1]; e.append(cont("2. Consistência com o Censo Escolar", a))
    a = a[a.IN_PREENCHIMENTO_LP == 1]; e.append(cont("3. Prova de LP preenchida", a))
    a = aluno_valido(aluno); e.append(cont("4. Proficiência em LP calculável (TRI)", a))
    a = a[a.IN_PUBLICA == 1]; e.append(cont("5. Rede pública", a))
    a = a[a.ID_ESCOLA.isin(com_inse)]; e.append(cont("6. Com INSE (TS_ESCOLA)", a))
    e.append(cont("   6a. turma com questionário de professor", a[a.ID_TURMA.isin(p.ID_TURMA)]))
    e.append(cont("   6b. escola com questionário de diretor", a[a.ID_ESCOLA.isin(d.ID_ESCOLA)]))
    com_bloco = set(a[a.ID_TURMA.isin(p.ID_TURMA)].ID_ESCOLA) | (set(a.ID_ESCOLA) & set(d.ID_ESCOLA))
    a = a[a.ID_ESCOLA.isin(com_bloco)]; e.append(cont("7. Escola com ≥ 1 bloco de preditores", a))
    a = a[a.ID_ESCOLA.isin(pop.index)]; e.append(cont(f"8. n ≥ {n_min} alunos → BASE FINAL (linhas = escolas)", a))
    e.append(cont("Coorte interna (professor E diretor)", alunos_interna))
    return pd.DataFrame(e).set_index("etapa")
