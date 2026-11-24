"""Orquestrador do pipeline. Uso:

    uv run python -m saeb_alfab.pipeline --modo rapido      # valida o pipeline (números NÃO citáveis)
    uv run python -m saeb_alfab.pipeline --modo completo    # execução final (vai para o texto)
    uv run python -m saeb_alfab.pipeline --ate modelagem    # para numa etapa: base | modelagem | explicabilidade | robustez
    uv run python -m saeb_alfab.pipeline --apenas pinball   # só a base + a perda de quantil (análise complementar)
    uv run python -m saeb_alfab.pipeline --apenas pesos     # só a base + o diagnóstico dos pesos amostrais
    uv run python -m saeb_alfab.pipeline --apenas descritivas  # só a base + as análises descritivas
    uv run python -m saeb_alfab.pipeline --apenas ranking-pesos  # ranking de temas do SHAP sem × com peso amostral
    uv run python -m saeb_alfab.pipeline --apenas ablacao   # valor incremental dos blocos, com os índices por tema

Cada execução grava em reports/execucoes/<data>_<modo>_s<semente>/ (tabelas .csv/.tex, figuras, run_info.json
com checksums dos dados, versões e commit do git). Nada sobrescreve execuções anteriores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import base as B
from . import descritivas as DS
from . import dicionario as D
from . import etl
from . import figuras_shap as FS
from .avaliacao import (comparar_no_teste, competir, competir_quantilica, cv_repetida, params_busca,
                        selecionar_modelo_final)
from .config import RAIZ, Config
from .explicabilidade import (estabilidade, lime_vs_shap, ranking_temas_ponderado, shap_abs_por_escola, shap_por_tema,
                              shap_por_variavel)
from .modelos import LINEAR, MODELOS, X_de, particao, pipeline_de, selecionar_variaveis, usa_peso_n
from .robustez import (ablacao, alfabetizacao_por_rede, auditar_equidade, construir_base_turma, diagnostico_pesos,
                       sensibilidades, teste_proxies, tres_execucoes)

ETAPAS = ["base", "modelagem", "explicabilidade", "robustez"]
TRILHAS = {"prop": ("Y_PROP_ALFAB", "Proporção de alfabetizados"), "prof": ("Y_PROF_MEDIA", "Proficiência média")}


class Saida:
    def __init__(self, pasta: Path):
        self.pasta = pasta
        (pasta / "tabelas").mkdir(parents=True, exist_ok=True)
        (pasta / "figuras").mkdir(parents=True, exist_ok=True)

    def tabela(self, df: pd.DataFrame, nome: str, indice: bool = True) -> pd.DataFrame:
        df.to_csv(self.pasta / "tabelas" / f"{nome}.csv", index=indice, encoding="utf-8")
        try:
            (self.pasta / "tabelas" / f"{nome}.tex").write_text(
                df.to_latex(index=indice, float_format="%.3f", escape=True), encoding="utf-8")
        except Exception:  # noqa: BLE001 — LaTeX é conveniência, CSV é a fonte
            pass
        return df

    def figura(self, fig, nome: str):
        fig.savefig(self.pasta / "figuras" / f"{nome}.png", dpi=200, bbox_inches="tight")


def _relativo(v) -> str:
    """Caminhos gravados relativos à raiz do repositório (o run_info não expõe a máquina de quem executou)."""
    try:
        return str(Path(v).resolve().relative_to(RAIZ)) if isinstance(v, Path) else str(v)
    except ValueError:
        return str(v)


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True)
        suja = subprocess.run(["git", "status", "--porcelain", "src"], cwd=RAIZ, capture_output=True, text=True).stdout
        return r.stdout.strip() + ("-modificado" if suja.strip() else "")
    except Exception:  # noqa: BLE001
        return "sem-git"


def executar(cfg: Config, ate: str = "robustez", apenas: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Avisos de convergência NÃO são silenciados globalmente; só os ruidosos e inofensivos.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
    t0 = time.time()
    carimbo = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Saida(cfg.saida_dir / f"{carimbo}_{cfg.modo}_s{cfg.seed}")
    info = {"inicio": dt.datetime.now().isoformat(timespec="seconds"), "modo": cfg.modo, "semente": cfg.seed,
            "citavel_no_texto": not cfg.rapido, "git": _git_commit(), "config": {k: _relativo(v) for k, v in cfg.__dict__.items()}}

    # ---------------------------------------------------------------- 1. base
    origem = etl.baixar_saeb(cfg.data_dir)
    info["checksums_ok"] = etl.verificar_checksums(cfg.data_dir)
    aluno, escola, prof, dire, planilhas = etl.ler_bases(cfg.data_dir)
    meta = D.montar_meta(D.parse_dicionario(planilhas["TS_PROFESSOR"]), D.parse_dicionario(planilhas["TS_DIRETOR"]))
    out.tabela(D.meta_para_csv(meta), "00_meta_variaveis", indice=False)
    base_todas, alunos_validos, prof_cod_turma, dir_cod_escola = B.construir_base(aluno, escola, prof, dire, meta,
                                                                                    cfg.juncao, 1)
    pop, universo = B.populacoes(base_todas, cfg.n_min_alunos)
    base_interna, alunos_interna, _, _ = B.construir_base(aluno, escola, prof, dire, meta, "interna", 1)
    base_interna = base_interna[(base_interna.PUBLICA == 1) & base_interna.TEM_INSE]
    alunos_interna = alunos_interna[alunos_interna.ID_ESCOLA.isin(base_interna.index)]
    out.tabela(B.funil(aluno, escola, prof, dire, pop, alunos_interna, cfg.n_min_alunos), "01_funil")
    pub = base_todas[(base_todas.PUBLICA == 1) & (base_todas.N_ALUNOS >= cfg.n_min_alunos)]
    cobertura = pd.DataFrame({
        "escolas públicas (n ≥ mínimo)": [len(pub)], "com INSE": [int(pub.TEM_INSE.sum())],
        "cobertura INSE": [pub.TEM_INSE.mean()], "pop: com questionário professor": [pop.TEM_PROF.mean()],
        "pop: com questionário diretor": [pop.TEM_DIR.mean()]}).T.rename(columns={0: "valor"})
    out.tabela(cobertura, "01_cobertura_blocos")
    ms_A, dec_A = selecionar_variaveis(pop, meta, com_cor=False, limite_ausencia=cfg.limite_ausencia,
                                       excluir_percepcao=cfg.excluir_percepcao_desfecho)
    ms_B, _ = selecionar_variaveis(pop, meta, com_cor=True, limite_ausencia=cfg.limite_ausencia,
                                   excluir_percepcao=cfg.excluir_percepcao_desfecho)
    out.tabela(dec_A.drop(columns=["mapa"]), "01_decisoes_variaveis_cenarioA", indice=False)
    info.update(origem_dados=origem, escolas_pop=len(pop), escolas_universo=len(universo), variaveis_A=len(ms_A),
                variaveis_B=len(ms_B))
    print(f"[base] pop={len(pop)} escolas · universo={len(universo)} · variáveis A={len(ms_A)} B={len(ms_B)}")
    _pesos(out, pop, aluno)
    _descritivas(out, pop, base_todas, alunos_validos, alunos_interna, dec_A, prof_cod_turma, dir_cod_escola, cfg)
    if ate == "base" or apenas in ("pesos", "descritivas"):
        return _fechar(out, info, t0)

    # ---------------------------------------------------------------- 2. modelagem (cenário A e B)
    idx_tr, idx_te = particao(pop, "Y_PROP_ALFAB", cfg.seed, cfg.teste)
    info.update(n_treino=len(idx_tr), n_teste=len(idx_te))
    if apenas == "pinball":
        info["apenas"] = "pinball"
        for tr, (alvo, rot) in TRILHAS.items():
            _pinball(out, pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
            print(f"[pinball] {rot}: concluída")
        return _fechar(out, info, t0)
    if apenas == "ablacao":
        info["apenas"] = "ablacao"
        for tr, (alvo, rot) in TRILHAS.items():
            res, aj, _ = competir(pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
            sel = selecionar_modelo_final(pop, ms_A, alvo, tr, idx_tr, aj, res, cfg)
            nome = sel["selecionado"]
            out.tabela(ablacao(pop, ms_A, alvo, tr, nome, params_busca(aj[nome]), cfg), f"04_ablacao_{tr}")
            print(f"[ablacao] {rot}: selecionado = {nome}")
        return _fechar(out, info, t0)
    if apenas == "figuras-shap":
        info["apenas"] = "figuras-shap"
        for tr, (alvo, rot) in TRILHAS.items():
            res, aj, _ = competir(pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
            sel = selecionar_modelo_final(pop, ms_A, alvo, tr, idx_tr, aj, res, cfg)
            pipe = aj[sel["selecionado"]].best_estimator_
            lime_tab, det = lime_vs_shap(pipe, pop, ms_A, idx_tr, idx_te, alvo, cfg,
                                         n_sementes=3 if cfg.rapido else 10, num_samples=2000 if cfg.rapido else 5000)
            out.tabela(lime_tab, f"03_lime_vs_shap_{tr}")
            _figuras_shap(out, pipe, pop, ms_A, idx_te, idx_tr, det, tr, rot)
            print(f"[figuras-shap] {rot}: selecionado = {sel['selecionado']}")
        return _fechar(out, info, t0)
    if apenas == "ranking-pesos":
        info["apenas"] = "ranking-pesos"
        for tr, (alvo, rot) in TRILHAS.items():
            res, aj, _ = competir(pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
            sel = selecionar_modelo_final(pop, ms_A, alvo, tr, idx_tr, aj, res, cfg)
            _ranking_pesos(out, pop, ms_A, tr, aj[sel["selecionado"]].best_estimator_, idx_tr, idx_te, cfg)
            print(f"[ranking-pesos] {rot}: selecionado = {sel['selecionado']}")
        return _fechar(out, info, t0)
    R = {}
    for tr, (alvo, rot) in TRILHAS.items():
        res, aj, pr = competir(pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
        out.tabela(res, f"02_competicao_{tr}_cenarioA")
        sel = selecionar_modelo_final(pop, ms_A, alvo, tr, idx_tr, aj, res, cfg)
        y_te = pop.loc[idx_te, alvo].values
        sel.update({f"teste_{k}": v for k, v in comparar_no_teste(y_te, pr[sel["melhor_ensemble"]], pr[LINEAR],
                                                                  cfg.n_boot, cfg.seed).items()})
        out.tabela(pd.Series(sel, name=rot).to_frame(), f"02_selecao_modelo_final_{tr}")
        resB, ajB, prB = competir(pop, ms_B, alvo, tr, idx_tr, idx_te, cfg)
        out.tabela(resB, f"02_competicao_{tr}_cenarioB")
        _pinball(out, pop, ms_A, alvo, tr, idx_tr, idx_te, cfg)
        R[tr] = dict(res=res, aj=aj, pr=pr, sel=sel, resB=resB, ajB=ajB)
        print(f"[modelagem] {rot}: selecionado = {sel['selecionado']} (ensemble candidato: {sel['melhor_ensemble']}; "
              f"Δ={sel['media']:.4f}, IC=[{sel['ic_inf']:.4f}; {sel['ic_sup']:.4f}])")
        fig, ax = plt.subplots(figsize=(5, 4.4))
        p_sel = pr[sel["selecionado"]]
        ax.scatter(y_te, p_sel, s=14, alpha=.6)
        lim = [min(y_te.min(), p_sel.min()), max(y_te.max(), p_sel.max())]
        ax.plot(lim, lim, "k--", lw=.8)
        ax.set(title=f"{rot}: observado × previsto (teste)\n{sel['selecionado']}", xlabel="observado", ylabel="previsto")
        out.figura(fig, f"02_observado_previsto_{tr}")
        plt.close(fig)
    if ate == "modelagem":
        return _fechar(out, info, t0)

    # ---------------------------------------------------------------- 3. explicabilidade (modelo SELECIONADO)
    for tr, (alvo, rot) in TRILHAS.items():
        nome = R[tr]["sel"]["selecionado"]
        pipe = R[tr]["aj"][nome].best_estimator_
        imp = shap_por_variavel(pipe, pop, ms_A, idx_te, idx_tr)
        tema = shap_por_tema(imp, ms_A)
        desc = ms_A.set_index("variavel")[["tema", "descricao", "percepcao_desfecho"]]
        out.tabela(imp.head(25).rename("participação").to_frame().join(desc), f"03_shap_variaveis_{tr}")
        out.tabela(tema, f"03_shap_temas_{tr}")
        _ranking_pesos(out, pop, ms_A, tr, pipe, idx_tr, idx_te, cfg)
        out.tabela(estabilidade(pop, ms_A, alvo, tr, nome, params_busca(R[tr]["aj"][nome]), cfg),
                   f"03_estabilidade_temas_{tr}")
        lime_tab, det = lime_vs_shap(pipe, pop, ms_A, idx_tr, idx_te, alvo, cfg,
                                     n_sementes=3 if cfg.rapido else 10, num_samples=2000 if cfg.rapido else 5000)
        out.tabela(lime_tab, f"03_lime_vs_shap_{tr}")
        for rotulo, df in det.items():
            out.tabela(df, f"03_lime_detalhe_{tr}_{FS.slug_caso(rotulo)}")
        _figuras_shap(out, pipe, pop, ms_A, idx_te, idx_tr, det, tr, rot)
        print(f"[explicabilidade] {rot}: top-3 temas = {', '.join(tema.index[:3])}")
    if ate == "explicabilidade":
        return _fechar(out, info, t0)

    # ---------------------------------------------------------------- 4. robustez e valor incremental
    base_turma = construir_base_turma(alunos_validos, prof_cod_turma, pop)
    out.tabela(teste_proxies(pop, ms_A, cfg), "04_teste_proxies")
    for tr, (alvo, rot) in TRILHAS.items():
        nome = R[tr]["sel"]["selecionado"]
        par = params_busca(R[tr]["aj"][nome])
        out.tabela(ablacao(pop, ms_A, alvo, tr, nome, par, cfg), f"04_ablacao_{tr}")
        out.tabela(tres_execucoes(pop, universo, meta, alvo, tr, nome, par, cfg), f"04_tres_execucoes_{tr}")
        out.tabela(sensibilidades(pop, base_todas, base_interna, base_turma, meta, alvo, tr, nome, par, cfg),
                   f"04_sensibilidades_{tr}")
        oa = cv_repetida(pop, ms_A, alvo, tr, nome, par, cfg)["pred"]
        ob = cv_repetida(pop, ms_B, alvo, tr, nome, params_busca(R[tr]["ajB"][nome]), cfg)["pred"]
        g, gaps = auditar_equidade(pop, alvo, oa, ob, cfg)
        out.tabela(g, f"04_equidade_grupos_{tr}")
        out.tabela(gaps, f"04_equidade_gaps_{tr}")
        print(f"[robustez] {rot}: concluída")
    return _fechar(out, info, t0)


def _ranking_pesos(out: Saida, pop, ms, tr, pipe, idx_tr, idx_te, cfg: Config) -> None:
    """Ranking de temas sem × com peso amostral nas 604 escolas (o teste sozinho tem poucas escolas
    efetivas). Checagem: o ranking sem peso nas 604 escolas deve acompanhar o oficial, do teste."""
    from scipy.stats import spearmanr
    tab, resumo = ranking_temas_ponderado(shap_abs_por_escola(pipe, pop, ms, pop.index, idx_tr), ms,
                                          pop["PESO"].values, cfg.n_boot, cfg.seed)
    oficial = shap_por_tema(shap_por_variavel(pipe, pop, ms, idx_te, idx_tr), ms)["participacao"]
    resumo.loc["Spearman sem peso: 604 escolas × teste"] = float(
        spearmanr(tab["participação sem peso"], oficial.reindex(tab.index).fillna(0)).statistic)
    out.tabela(tab, f"03_shap_temas_ponderado_{tr}")
    out.tabela(resumo, f"03_shap_temas_ponderado_resumo_{tr}")


def _figuras_shap(out: Saida, pipe, pop, ms, idx_te, idx_tr, det, tr, rot) -> None:
    import matplotlib.pyplot as plt
    for nome, fig in FS.gerar(pipe, pop, ms, idx_te, idx_tr, det, rot).items():
        out.figura(fig, f"03_shap_{nome}_{tr}")
        plt.close(fig)


def _descritivas(out: Saida, pop, base_todas, alunos_validos, alunos_interna, decisoes, prof_cod, dir_cod,
                 cfg: Config) -> None:
    """Descritivas do capítulo de Resultados (sem peso e com o peso amostral da escola, decisão D2)."""
    out.tabela(DS.tabela_unidade(alunos_interna, alunos_validos, pop), "01_unidade_analise")
    out.tabela(DS.estatisticas_desfechos(pop), "01_desfechos")
    out.tabela(DS.por_geografia(pop), "01_desfechos_por_geografia")
    niveis, resumo = DS.por_inse(pop)
    out.tabela(niveis, "01_desfechos_por_inse")
    out.tabela(resumo, "01_inse_sozinho")
    out.tabela(DS.funil_binomial(pop), "01_funil_binomial")
    cob = DS.cobertura_por_variavel(pop, decisoes, prof_cod, dir_cod)
    out.tabela(cob, "01_cobertura_por_variavel")
    out.tabela(DS.cobertura_por_tema(cob), "01_cobertura_por_tema")
    out.tabela(DS.ausencia_informativa(pop), "01_ausencia_informativa")
    out.tabela(DS.vies_inse(base_todas, cfg.n_min_alunos), "01_vies_selecao_inse")
    out.tabela(DS.composicao_cor(pop), "01_cor_composicao")
    bruta, estrato = DS.associacao_cor(pop)
    out.tabela(bruta, "01_cor_associacao_bruta")
    out.tabela(estrato, "01_cor_por_faixa_de_inse")
    import matplotlib.pyplot as plt
    for nome, fig in DS.figuras(pop).items():
        out.figura(fig, nome)
        plt.close(fig)


def _pesos(out: Saida, pop, aluno) -> None:
    resumo, medias = diagnostico_pesos(pop)
    out.tabela(resumo, "01_pesos_diagnostico")
    out.tabela(medias, "01_pesos_medias")
    out.tabela(alfabetizacao_por_rede(aluno), "01_alfabetizacao_por_rede")


def _pinball(out: Saida, pop, ms, alvo, tr, idx_tr, idx_te, cfg: Config) -> None:
    perdas, ic, diag, cruz = competir_quantilica(pop, ms, alvo, tr, idx_tr, idx_te, cfg)
    out.tabela(perdas, f"02_pinball_{tr}")
    out.tabela(ic, f"02_pinball_ic_{tr}")
    out.tabela(diag, f"02_pinball_diagnostico_{tr}")
    out.tabela(cruz, f"02_pinball_cruzamento_{tr}")


def _fechar(out: Saida, info: dict, t0: float) -> Path:
    import lightgbm, shap, sklearn, xgboost  # noqa: E401

    info.update(duracao_min=round((time.time() - t0) / 60, 1), python=sys.version.split()[0],
                sistema=platform.platform(),
                pacotes={"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__})
    (out.pasta / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[ok] {out.pasta} ({info['duracao_min']} min)")
    return out.pasta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["rapido", "completo"], default=None)
    ap.add_argument("--ate", choices=ETAPAS, default="robustez")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--apenas", choices=["descritivas", "pinball", "pesos", "ranking-pesos", "ablacao", "figuras-shap"],
                    default=None)
    a = ap.parse_args(argv)
    cfg = Config()
    if a.modo:
        cfg = replace(cfg, modo=a.modo)
    if a.seed is not None:
        cfg = replace(cfg, seed=a.seed)
    executar(cfg, a.ate, a.apenas)


if __name__ == "__main__":
    main()
