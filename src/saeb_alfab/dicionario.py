"""Leitura do dicionário do Inep, classificação dos itens e codificação.

CORREÇÃO EM RELAÇÃO À VERSÃO DO TCC 1: o parser anterior ignorava a coluna
`TIPO` do dicionário e inferia o tipo por palavras-chave; com isso, 7 itens numéricos
(idade, anos de docência/direção) viravam nominais (one-hot + moda) e as escalas com
"Não tem/Não uso" viravam ordinais pela POSIÇÃO DA LETRA ("Não tem"=1 < "Muito inadequado").

Agora:
* `TIPO == NUM`  -> numérico (média na agregação turma -> escola);
* escalas ordinais são reconhecidas pelo TEXTO das opções (não pela letra), com ordem
  explícita em `ESCALAS`;
* opções fora da escala ("Não tem", "Não uso", "Não foi realizada esta ação", "Não se
  aplica", "Não sei") viram um INDICADOR binário próprio (`<var>__FORA_ESCALA`) e NaN na
  escala — a informação não se perde e não é ordenada artificialmente;
* toda decisão (tipo, mapa, tema, exclusão) fica em `meta`, exportado como CSV versionado.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


def norm(txt) -> str:
    """Normaliza o texto de uma opção: minúsculas, sem ponto final, espaços colapsados."""
    s = unicodedata.normalize("NFC", str(txt)).strip().lower().rstrip(".").strip()
    return re.sub(r"\s+", " ", s)


# ------------------------------------------------------------------ escalas ordinais (ordem explícita)
ESCALAS: dict[str, list[str]] = {
    "frequencia": ["nunca", "poucas vezes", "muitas vezes", "sempre"],
    "concordancia": ["discordo fortemente", "discordo", "concordo", "concordo fortemente"],
    "adequacao": ["muito inadequado", "inadequado", "adequado", "muito adequado"],
    "necessidade": ["nenhuma necessidade", "pouca necessidade", "moderada necessidade", "muita necessidade"],
    "contribuicao": ["não contribuiu", "contribuiu pouco", "contribuiu razoavelmente", "contribuiu muito"],
    "efetividade": ["nada efetiva", "pouco efetiva", "efetiva", "muito efetiva"],
    "contagem_0a3": ["nenhuma", "uma", "duas", "três ou mais"],
    "vezes_1a3": ["uma vez", "duas vezes", "três vezes ou mais"],
    "apoio": ["sem apoio", "com apoio parcial", "com apoio total"],
    "atividades": ["nenhuma atividade formativa", "algumas atividades formativas (ou parte delas)",
                   "todas as atividades formativas"],
    "n_escolas": ["apenas nesta", "em 2", "em 3 ou mais"],
    "escolaridade_prof": ["ensino médio - magistério", "graduação", "especialização", "mestrado", "doutorado"],
    "escolaridade_dir": ["ensino fundamental", "ensino médio", "graduação", "especialização", "mestrado", "doutorado"],
    "pos_graduacao": ["não fiz curso de pós-graduação", "especialização (mínimo de 360 horas)",
                      "mestrado (acadêmico ou profissional)", "doutorado"],
}
FORA_DE_ESCALA = {"não tem", "não uso", "não foi realizada esta ação", "não se aplica", "não sei"}
POSITIVOS_BINARIOS = {"sim", "suficiente", "ativo"}
FAIXA_SALARIAL = re.compile(r"r\$")


# ------------------------------------------------------------------ grupos temáticos (herdados do TCC 1)
TEMA_RACA = "Cor/raça do respondente (sensível, LGPD)"
TEMAS_PROF = [
    (1, 2, "Perfil demográfico do professor"), (3, 3, TEMA_RACA),
    (4, 7, "Saúde/deficiência (sensível — excluído sempre)"),
    (8, 14, "Hábitos culturais e lazer do professor"), (15, 19, "Satisfação e identidade profissional"),
    (20, 20, "Formação e escolaridade do professor"), (21, 33, "Formação continuada (curta duração)"),
    (34, 36, "Pós-graduação do professor"), (37, 47, "Necessidade percebida de formação continuada"),
    (48, 57, "Experiência, vínculo e carga horária"), (58, 75, "Infraestrutura da sala de aula (percepção)"),
    (76, 80, "Crenças pedagógicas (repetência, avaliações externas)"),
    (81, 96, "Práticas pedagógicas em sala de aula"), (97, 99, "Educação especial (AEE) na turma"),
    (100, 108, "Gestão escolar, PPP e conselho de classe (visão do professor)"),
    (109, 114, "Autonomia pedagógica"), (115, 126, "Liderança do diretor e colaboração (visão do professor)"),
    (127, 134, "Clima escolar — relação com estudantes"), (135, 147, "Clima escolar — segurança e violência"),
    (148, 148, "Texto aberto (excluído)"),
]
TEMAS_DIR = [
    (1, 5, "Caracterização da escola (etapas oferecidas)"), (6, 6, "Rede pública/privada (redundante, excluído)"),
    (7, 8, "Perfil demográfico do diretor"), (9, 9, TEMA_RACA),
    (10, 13, "Saúde/deficiência (sensível — excluído sempre)"),
    (14, 17, "Formação e carreira do diretor"), (18, 26, "Carga horária e condições de trabalho do diretor"),
    (27, 31, "Percepções pedagógicas do diretor"), (32, 37, "Infraestrutura — equipamentos gerais"),
    (38, 51, "Infraestrutura — Educação Infantil"), (52, 53, "Educação Infantil (esparso, excluído)"),
    (54, 54, "Infraestrutura — Educação Infantil"), (55, 67, "Recursos financeiros e de pessoal"),
    (68, 77, "Atendimento Educacional Especializado (AEE)"), (78, 88, "Interrupções do calendário"),
    (89, 89, "Texto aberto (excluído)"), (90, 90, "Interrupções do calendário"),
    (91, 103, "Segurança — episódios de violência"), (104, 109, "Segurança física e acessibilidade"),
    (110, 118, "Gestão democrática e colegiados"),
    (119, 120, "Caracterização institucional"), (121, 128, "Parcerias externas"),
    (129, 138, "Financiamento da escola"), (139, 150, "Merenda escolar"),
    (151, 159, "Projeto Político-Pedagógico"), (160, 167, "Critérios de matrícula"),
    (168, 168, "Texto aberto (excluído)"), (169, 185, "Composição de turmas e alocação de professores"),
    (186, 193, "Ações contra repetência e abandono"), (194, 204, "Projetos temáticos"),
    (205, 210, "Estímulo acadêmico e extracurricular"), (211, 211, "Texto aberto (excluído)"),
    (212, 223, "Formação continuada percebida (equipe)"), (224, 224, "Texto aberto (excluído)"),
]
# Percepção do respondente sobre o desempenho dos próprios alunos (risco de causalidade reversa).
PERCEPCAO_DESFECHO = {"PROF_TX_Q080", "DIR_TX_Q030"}


def tema_de(n: int, faixas) -> str:
    return next((t for a, b, t in faixas if a <= n <= b), "Sem tema")


# ------------------------------------------------------------------ parsing
def parse_dicionario(aba: pd.DataFrame) -> pd.DataFrame:
    """Bloco de itens `TX_Q…` de uma aba (lida com header=None) -> var, tipo, grupo, desc, cod, resp.
    O cabeçalho é a ÚLTIMA linha cuja 1ª célula é 'Variável' (robusto a mudanças de layout)."""
    primeira = aba.iloc[:, 0].astype(str).str.strip().str.lower()
    cab = [i for i, v in primeira.items() if v in ("variável", "variavel")]
    if not cab:
        raise ValueError("cabeçalho 'Variável' não encontrado na aba do dicionário")
    b = aba.loc[cab[-1]:].copy()
    b.columns = [norm(c) for c in b.iloc[0]]
    b = b.iloc[1:].reset_index(drop=True)
    ren = {}
    for c in b.columns:
        if c in ("variável", "variavel"):
            ren[c] = "var"
        elif c == "tipo":
            ren[c] = "tipo"
        elif c == "grupo":
            ren[c] = "grupo"
        elif c.startswith("descri"):
            ren[c] = "desc"
        elif c.startswith("código da op") or c.startswith("codigo da op"):
            ren[c] = "cod"
        elif c == "resposta":
            ren[c] = "resp"
    b = b.rename(columns=ren)
    faltam = {"var", "tipo", "desc", "cod", "resp"} - set(b.columns)
    if faltam:
        raise ValueError(f"colunas ausentes no dicionário: {faltam}")
    if "grupo" not in b:
        b["grupo"] = np.nan
    b = b[["var", "tipo", "grupo", "desc", "cod", "resp"]]
    b[["var", "tipo", "grupo", "desc"]] = b[["var", "tipo", "grupo", "desc"]].ffill()
    return b[b["var"].astype(str).str.startswith("TX_Q")].reset_index(drop=True)


def classificar_item(g: pd.DataFrame) -> dict:
    """Classifica um item a partir de TIPO e do TEXTO das opções.
    Devolve dict(tipo, escala, mapa) em que `mapa` leva letra -> valor (ou 'FORA' para fora da escala)."""
    tipo_inep = str(g["tipo"].dropna().iloc[0]).strip().upper() if g["tipo"].notna().any() else ""
    if tipo_inep == "NUM":
        return dict(tipo="numerico", escala=None, mapa=None)
    letras = [str(c).strip().upper() for c in g["cod"]]
    ops = [norm(r) for r in g["resp"]]
    letra_txt = dict(zip(letras, ops))
    if any(o == "resposta aberta" for o in ops):
        return dict(tipo="texto", escala=None, mapa=None)
    if len(ops) == 1:                                     # caixa de seleção única (ex.: "Nunca trabalhei")
        return dict(tipo="marcacao", escala=None, mapa={letras[0]: 1.0})
    na_escala = [o for o in ops if o not in FORA_DE_ESCALA]
    tem_fora = len(na_escala) < len(ops)
    if len(na_escala) == 2 and set(na_escala) & (POSITIVOS_BINARIOS | {"não", "insuficiente", "inativo"}):
        mapa = {l: ("FORA" if t in FORA_DE_ESCALA else float(t in POSITIVOS_BINARIOS)) for l, t in letra_txt.items()}
        return dict(tipo="binario", escala="sim_nao" + ("+fora" if tem_fora else ""), mapa=mapa)
    for nome, ordem in ESCALAS.items():
        if na_escala in (ordem, ordem[::-1]):            # aceita a escala listada em ordem decrescente
            base0 = nome in ("contagem_0a3",)
            mapa = {l: ("FORA" if t in FORA_DE_ESCALA else float(ordem.index(t) + (0 if base0 else 1)))
                    for l, t in letra_txt.items()}
            return dict(tipo="ordinal", escala=nome + ("+fora" if tem_fora else ""), mapa=mapa)
    if all(FAIXA_SALARIAL.search(o) for o in ops):        # faixas salariais já vêm em ordem crescente
        return dict(tipo="ordinal", escala="faixa_salarial", mapa={l: float(i + 1) for i, l in enumerate(letras)})
    return dict(tipo="nominal", escala=None, mapa={l: t for l, t in letra_txt.items()})


def montar_meta(dic_prof: pd.DataFrame, dic_dir: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por variável candidata (já incluindo os indicadores `__FORA_ESCALA` derivados)."""
    linhas = []
    for pref, dic, faixas in (("PROF", dic_prof, TEMAS_PROF), ("DIR", dic_dir, TEMAS_DIR)):
        for var, g in dic.groupby("var", sort=False):
            n = int(re.match(r"TX_Q(\d+)", var).group(1))
            tema = tema_de(n, faixas)
            if tema == TEMA_RACA:
                continue                                   # substituída por variáveis derivadas (base.py)
            c = classificar_item(g)
            nome = f"{pref}_{var}"
            excluir = ("excluído" in tema) or c["tipo"] == "texto"
            comum = dict(origem=pref, item=var, tema=tema, grupo_inep=str(g["grupo"].iloc[0]).strip(),
                         descricao=str(g["desc"].iloc[0]).strip(), sensivel=False, excluir_sempre=excluir,
                         percepcao_desfecho=nome in PERCEPCAO_DESFECHO)
            tipo = "binario" if c["tipo"] == "marcacao" else c["tipo"]
            linhas.append(dict(variavel=nome, tipo=tipo, escala=c["escala"], mapa=c["mapa"], derivada_de=None, **comum))
            if c["escala"] and c["escala"].endswith("+fora"):
                linhas.append(dict(variavel=f"{nome}__FORA_ESCALA", tipo="binario", escala="indicador_fora_escala",
                                   mapa=None, derivada_de=nome, **comum))
    extras = [
        ("GEO_ID_REGIAO", "Contexto geográfico", "nominal", "Região do Brasil"),
        ("GEO_ID_UF", "Contexto geográfico", "nominal", "Unidade da Federação"),
        ("GEO_ID_AREA", "Contexto geográfico", "nominal", "Capital ou interior"),
        ("GEO_ID_LOCALIZACAO", "Contexto geográfico", "nominal", "Urbana ou rural"),
        ("ESC_INSE", "Nível socioeconômico (INSE)", "ordinal", "INSE da escola (1 = Nível I … 8 = Nível VIII)"),
        ("ESC_PC_FORMACAO_DOCENTE_INICIAL", "Formação docente (escola)", "numerico",
         "% de docentes com formação adequada (anos iniciais)"),
        ("CTX_N_TURMAS", "Contexto de agregação", "numerico", "Nº de turmas de 2º ano com questionário de professor"),
        ("PROF_COR_PCT_PRETA_PARDA", TEMA_RACA, "numerico", "% das turmas cujo professor se declara preto/pardo"),
        ("PROF_COR_PCT_BRANCA", TEMA_RACA, "numerico", "% das turmas cujo professor se declara branco"),
        ("PROF_COR_PCT_OUTRA_ND", TEMA_RACA, "numerico", "% das turmas cujo professor é amarelo/indígena ou não declara"),
        ("DIR_COR", TEMA_RACA, "nominal", "Cor/raça declarada do diretor (agrupada)"),
    ]
    for v, t, tp, d in extras:
        linhas.append(dict(variavel=v, origem=v.split("_")[0], item=v, tema=t, grupo_inep=None, tipo=tp, escala=None,
                           mapa=None, derivada_de=None, descricao=d, sensivel=(t == TEMA_RACA), excluir_sempre=False,
                           percepcao_desfecho=False))
    meta = pd.DataFrame(linhas)
    assert meta.variavel.is_unique, "nomes de variáveis duplicados no meta"
    return meta


# ------------------------------------------------------------------ codificação
def codificar_item(serie: pd.Series, linha: pd.Series) -> dict[str, pd.Series]:
    """Codifica uma coluna bruta segundo a linha do meta. Pode devolver 2 colunas (escala + indicador)."""
    tipo, mapa, nome = linha["tipo"], linha["mapa"], linha["variavel"]
    if tipo == "texto":
        return {}                                          # resposta aberta: nunca vira preditor
    if tipo == "numerico":
        return {nome: pd.to_numeric(serie, errors="coerce")}
    s = serie.astype("string").str.strip().str.upper()
    if tipo == "nominal":
        return {nome: s.map(mapa).astype(object) if mapa else s.astype(object)}
    if linha["escala"] == "indicador_fora_escala":
        return {}                                          # gerado junto com a variável-mãe
    if mapa and len(mapa) == 1 and list(mapa.values()) == [1.0]:   # marcação única: marcado=1, vazio=0
        return {nome: s.notna().astype(float)}
    valores = s.map(mapa)
    fora = valores.eq("FORA")
    out = {nome: pd.to_numeric(valores.where(~fora), errors="coerce").astype(float)}
    if linha["escala"] and str(linha["escala"]).endswith("+fora"):
        ind = fora.astype(float).where(s.notna())          # NaN se o item não foi respondido
        out[f"{nome}__FORA_ESCALA"] = ind
    return out


def codificar_bloco(df: pd.DataFrame, meta: pd.DataFrame, origem: str) -> pd.DataFrame:
    m = meta[(meta.origem == origem) & meta.item.str.startswith("TX_Q") & meta.derivada_de.isna()]
    cols: dict[str, pd.Series] = {}
    for _, r in m.iterrows():
        if r["item"] in df.columns:
            cols.update(codificar_item(df[r["item"]], r))
    return pd.DataFrame(cols, index=df.index)


def meta_para_csv(meta: pd.DataFrame) -> pd.DataFrame:
    """Versão serializável do meta (mapas como texto) — a 'fonte única de verdade' prometida no texto."""
    out = meta.copy()
    out["mapa"] = out["mapa"].map(lambda m: "; ".join(f"{k}→{v}" for k, v in m.items()) if isinstance(m, dict) else "")
    return out
