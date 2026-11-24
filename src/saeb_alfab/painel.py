"""Protótipo do painel: só agregados, com supressão de células pequenas (LGPD, art. 12).

Duas partes:
- ``agregados``: tabelas por UF, região, localização e faixa de INSE, com desfechos sem e com peso amostral e o
  resíduo médio do modelo selecionado (observado − previsto fora da amostra). Grupos com menos de ``n_min`` escolas
  têm todos os valores suprimidos. Nenhuma escola individual aparece.
- ``gerar_html``: página autônoma (HTML, CSS e JavaScript inline, sem bibliotecas externas) a partir dessas tabelas
  e das tabelas de estabilidade dos temas.

Uso: python -m saeb_alfab.painel <pasta_painel> <pasta_tabelas_finais> <saida.html>
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


def _resumo(g: pd.DataFrame) -> pd.Series:
    d = {"escolas": len(g), "alunos": int(g.N_ALUNOS.sum()), "INSE médio": g.ESC_INSE.mean()}
    for y, k in DESFECHOS.items():
        d[f"{k} sem peso"] = g[y].mean()
        d[f"{k} com peso"] = _media_pond(g[y], g.PESO)
        d[f"{k} resíduo médio"] = g[f"_res_{k}"].mean()
        d[f"{k} erro absoluto médio"] = g[f"_res_{k}"].abs().mean()
    return pd.Series(d)


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


def gerar_html(pasta_painel: Path, pasta_tabelas: Path, n_min: int, versao: str) -> str:
    dados = {k: _registros(pd.read_csv(pasta_painel / f"05_painel_{k}.csv", index_col=0))
             for k in ("uf", "regiao", "localizacao", "inse", "total")}
    for k in ("uf", "regiao", "localizacao", "inse", "total"):
        for r in dados[k]:
            r["suprimido"] = bool(r.get("suprimido"))
    temas = {}
    for tr in ("prop", "prof"):
        e = pd.read_csv(pasta_tabelas / f"03_estabilidade_temas_{tr}.csv", index_col=0).head(10)
        temas[tr] = [{"tema": i, "participacao": float(r["participação média"]), "top3": float(r["% repetições no top-3"]),
                      "posicao": float(r["posição média"])} for i, r in e.iterrows()]
    meta = {"n_min": n_min, "versao": versao}
    return _MODELO.replace("/*DADOS*/", json.dumps({"dados": dados, "temas": temas, "meta": meta}, ensure_ascii=False))


_MODELO = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Painel da alfabetização no 2º ano</title>
<style>
:root{--fundo:#f7f6f2;--cartao:#fff;--tinta:#1b1b1a;--tinta2:#5a5955;--grade:#e2e0da;--azul:#2a78d6;--laranja:#eb6834;--neg:#b8462a}
*{box-sizing:border-box}body{margin:0;background:var(--fundo);color:var(--tinta);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:28px 16px 8px;max-width:1080px;margin:auto}h1{font-size:24px;margin:0 0 4px}header p{margin:0;color:var(--tinta2)}
nav{display:flex;gap:6px;flex-wrap:wrap;max-width:1080px;margin:16px auto 0;padding:0 16px}
nav button{border:1px solid var(--grade);background:var(--cartao);padding:8px 14px;border-radius:999px;cursor:pointer;font:inherit;color:var(--tinta)}
nav button[aria-selected=true]{background:var(--tinta);color:#fff;border-color:var(--tinta)}
main{max-width:1080px;margin:auto;padding:16px}section{display:none}section.ativo{display:block}
.cartao{background:var(--cartao);border:1px solid var(--grade);border-radius:12px;padding:18px;margin-bottom:16px}
.cartao h2{font-size:17px;margin:0 0 6px}.nota{color:var(--tinta2);font-size:13px}
.controles{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0 12px}
select,label{font:inherit}select{padding:6px 8px;border-radius:8px;border:1px solid var(--grade);background:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--cartao);border:1px solid var(--grade);border-radius:12px;padding:14px}.kpi b{display:block;font-size:24px}
.kpi span{color:var(--tinta2);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:6px 8px;border-bottom:1px solid var(--grade);text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--tinta2);font-weight:600}
.tabela{overflow-x:auto}.sup{color:var(--tinta2);font-style:italic}
svg text{font:12px system-ui,sans-serif;fill:var(--tinta)}svg .eixo{fill:var(--tinta2)}
.regras li{margin-bottom:6px}footer{max-width:1080px;margin:auto;padding:8px 16px 32px;color:var(--tinta2);font-size:12.5px}
</style>
</head>
<body>
<header>
<h1>Alfabetização no 2º ano: escolas públicas do Saeb 2023</h1>
<p>Protótipo de painel com dados agregados das 604 escolas analisadas no trabalho. Nenhuma escola individual é exibida.</p>
</header>
<nav role="tablist">
<button role="tab" aria-selected="true" data-aba="geo">Por unidade da federação</button>
<button role="tab" aria-selected="false" data-aba="inse">Por nível socioeconômico</button>
<button role="tab" aria-selected="false" data-aba="temas">Temas do modelo</button>
<button role="tab" aria-selected="false" data-aba="erro">Onde o modelo erra</button>
<button role="tab" aria-selected="false" data-aba="regras">Como ler</button>
</nav>
<main>
<section id="geo" class="ativo">
<div class="kpis" id="kpis"></div>
<div class="cartao">
<h2>Desfechos por unidade da federação</h2>
<div class="controles">
<label>Desfecho <select id="g-desf"><option value="prop">Proporção de alfabetizados</option><option value="prof">Proficiência média em LP</option></select></label>
<label>Região <select id="g-reg"><option value="">Todas</option></select></label>
<label><input type="checkbox" id="g-peso" checked> com peso amostral</label>
</div>
<div id="g-graf"></div>
<p class="nota" id="g-nota"></p>
<div class="tabela"><table id="g-tab"></table></div>
</div>
</section>
<section id="inse">
<div class="cartao">
<h2>Desfechos por faixa de INSE da escola</h2>
<div class="controles"><label>Desfecho <select id="i-desf"><option value="prop">Proporção de alfabetizados</option><option value="prof">Proficiência média em LP</option></select></label></div>
<div id="i-graf"></div>
<p class="nota">Azul: média simples das escolas. Laranja: média com o peso amostral. Com peso, as faixas baixas reúnem poucas escolas efetivas e a média é imprecisa.</p>
</div>
</section>
<section id="temas">
<div class="cartao">
<h2>Temas de maior participação na previsão (SHAP do modelo selecionado)</h2>
<div class="controles"><label>Desfecho <select id="t-desf"><option value="prop">Proporção de alfabetizados</option><option value="prof">Proficiência média em LP</option></select></label></div>
<div id="t-graf"></div>
<p class="nota">Barra: participação média do tema na importância total em 20 partições. Número à direita: porcentagem das partições em que o tema ficou entre os três primeiros. Só ordenações estáveis devem ser interpretadas. Participação na previsão indica associação e não mede efeito causal.</p>
</div>
</section>
<section id="erro">
<div class="cartao">
<h2>Resíduo médio por grupo de escolas</h2>
<div class="controles">
<label>Desfecho <select id="e-desf"><option value="prop">Proporção de alfabetizados</option><option value="prof">Proficiência média em LP</option></select></label>
<label>Grupos <select id="e-eixo"><option value="inse">Faixa de INSE</option><option value="regiao">Região</option><option value="localizacao">Localização</option></select></label>
</div>
<div id="e-graf"></div>
<p class="nota">Resíduo: desfecho observado menos o previsto pelo modelo fora da amostra. Positivo: o grupo fica, em média, acima do que o modelo prevê para o seu contexto. O erro absoluto médio mostra onde a previsão é menos precisa.</p>
<div class="tabela"><table id="e-tab"></table></div>
</div>
</section>
<section id="regras">
<div class="cartao regras">
<h2>Como ler este painel</h2>
<ul>
<li>Os números descrevem as 604 escolas públicas com INSE e pelo menos 10 alunos avaliados no 2º ano do Saeb 2023.</li>
<li>A média com peso amostral aproxima a rede representada pela amostra. Com 108 escolas efetivas, ela é imprecisa em grupos pequenos.</li>
<li>Grupos com menos de <span id="r-nmin"></span> escolas têm os valores suprimidos, para evitar a identificação de escolas.</li>
<li>Os temas e os resíduos vêm de um modelo preditivo. Eles indicam associações no nível da escola e não medem o efeito de mudar qualquer fator.</li>
<li>O modelo erra mais nas escolas de INSE baixo. Por isso o painel não classifica escolas individuais como acima ou abaixo do esperado.</li>
</ul>
</div>
</section>
</main>
<footer id="rodape"></footer>
<script>
const D=/*DADOS*/;
const AZUL="#2a78d6",LARANJA="#eb6834";
const fmt=(v,c)=>v==null||Number.isNaN(v)?"—":v.toLocaleString("pt-BR",{minimumFractionDigits:c,maximumFractionDigits:c});
const pct=v=>v==null?"—":fmt(100*v,1)+"%";
const val=(k,v)=>k==="prop"?pct(v):fmt(v,1);
const rot={prop:"Proporção de alfabetizados",prof:"Proficiência média em LP"};
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button").forEach(x=>x.setAttribute("aria-selected",x===b));document.querySelectorAll("section").forEach(s=>s.classList.toggle("ativo",s.id===b.dataset.aba));});
function barras(el,itens,o){ // itens: [{rot, v, v2?, dir?}]
  const h=26,top=8,esq=o.esq||170,larg=Math.max(320,Math.min(980,el.clientWidth||900)),dir=o.dirTxt?64:24,W=larg-esq-dir;
  const vals=itens.flatMap(i=>[i.v,i.v2]).filter(v=>v!=null);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(o.lo!=null)lo=o.lo;if(o.hi!=null)hi=o.hi;if(hi===lo)hi=lo+1;
  const x=v=>esq+(v-lo)/(hi-lo)*W,H=top+itens.length*h+24;let s=`<svg viewBox="0 0 ${larg} ${H}" width="100%" role="img" aria-label="${o.titulo||""}">`;
  s+=`<line x1="${x(0)}" x2="${x(0)}" y1="${top}" y2="${H-20}" stroke="#bdbab2"/>`;
  itens.forEach((it,i)=>{const y=top+i*h;s+=`<text x="${esq-8}" y="${y+h/2+4}" text-anchor="end">${it.rot}</text>`;
    if(it.v==null){s+=`<text x="${esq+4}" y="${y+h/2+4}" class="eixo">suprimido (menos de ${D.meta.n_min} escolas)</text>`;return;}
    const dupla=it.v2!=null,bh=dupla?9:16,y1=y+(h-(dupla?20:16))/2;
    s+=`<rect x="${Math.min(x(0),x(it.v))}" y="${y1}" width="${Math.abs(x(it.v)-x(0))}" height="${bh}" rx="2" fill="${it.cor||AZUL}"><title>${it.rot}: ${o.f(it.v)}</title></rect>`;
    if(dupla)s+=`<rect x="${Math.min(x(0),x(it.v2))}" y="${y1+11}" width="${Math.abs(x(it.v2)-x(0))}" height="${bh}" rx="2" fill="${LARANJA}"><title>${it.rot} (com peso): ${o.f(it.v2)}</title></rect>`;
    const xt=Math.max(x(it.v),dupla?x(it.v2):-1e9);s+=`<text x="${(it.v>=0?xt+6:x(0)+6)}" y="${y+h/2+4}">${o.dirTxt?o.dirTxt(it):o.f(it.v)}</text>`;});
  s+=`<text x="${esq}" y="${H-4}" class="eixo">${o.f(lo)}</text><text x="${esq+W}" y="${H-4}" text-anchor="end" class="eixo">${o.f(hi)}</text></svg>`;el.innerHTML=s;}
// KPIs
(()=>{const t=D.dados.total[0];document.getElementById("kpis").innerHTML=[["Escolas",fmt(t.escolas,0)],["Alunos avaliados",fmt(t.alunos,0)],["Alfabetizados (com peso)",pct(t["prop com peso"])],["Proficiência média (com peso)",fmt(t["prof com peso"],1)]].map(([a,b])=>`<div class="kpi"><span>${a}</span><b>${b}</b></div>`).join("");})();
// Geografia
const regs=[...new Set(D.dados.uf.map(r=>r["região"]).filter(Boolean))].sort();
regs.forEach(r=>document.getElementById("g-reg").insertAdjacentHTML("beforeend",`<option>${r}</option>`));
function geo(){const k=document.getElementById("g-desf").value,reg=document.getElementById("g-reg").value,peso=document.getElementById("g-peso").checked,col=`${k} ${peso?"com":"sem"} peso`;
  let linhas=D.dados.uf.filter(r=>!reg||r["região"]===reg||r.suprimido);
  linhas.sort((a,b)=>(b[col]??-1e9)-(a[col]??-1e9));
  barras(document.getElementById("g-graf"),linhas.map(r=>({rot:r.grupo,v:r.suprimido?null:r[col]})),{f:v=>val(k,v),esq:60,lo:k==="prof"?650:0,titulo:rot[k]});
  document.getElementById("g-nota").textContent=`${rot[k]} ${peso?"com":"sem"} peso amostral. UFs com menos de ${D.meta.n_min} escolas na base aparecem como suprimidas.`;
  document.getElementById("g-tab").innerHTML="<tr><th>UF</th><th>Região</th><th>Escolas</th><th>Alunos</th><th>INSE médio</th><th>Sem peso</th><th>Com peso</th></tr>"+linhas.map(r=>r.suprimido?`<tr><td>${r.grupo}</td><td></td><td colspan="5" class="sup">suprimido</td></tr>`:`<tr><td>${r.grupo}</td><td>${r["região"]}</td><td>${fmt(r.escolas,0)}</td><td>${fmt(r.alunos,0)}</td><td>${fmt(r["INSE médio"],1)}</td><td>${val(k,r[k+" sem peso"])}</td><td>${val(k,r[k+" com peso"])}</td></tr>`).join("");}
["g-desf","g-reg","g-peso"].forEach(i=>document.getElementById(i).onchange=geo);
// INSE
function inse(){const k=document.getElementById("i-desf").value;barras(document.getElementById("i-graf"),D.dados.inse.map(r=>({rot:`${r.grupo} (${r.suprimido?"—":fmt(r.escolas,0)} escolas)`,v:r.suprimido?null:r[k+" sem peso"],v2:r.suprimido?null:r[k+" com peso"]})),{f:v=>val(k,v),esq:170,lo:k==="prof"?650:0});}
document.getElementById("i-desf").onchange=inse;
// Temas
function temas(){const k=document.getElementById("t-desf").value;barras(document.getElementById("t-graf"),D.temas[k].map(t=>({rot:t.tema.length>44?t.tema.slice(0,43)+"…":t.tema,v:t.participacao,top3:t.top3,cor:t.top3>=75?AZUL:"#9ec5f4"})),{f:v=>pct(v),esq:300,dirTxt:it=>`${fmt(it.top3,0)}%`});}
document.getElementById("t-desf").onchange=temas;
// Erro
function erro(){const k=document.getElementById("e-desf").value,e=document.getElementById("e-eixo").value,linhas=D.dados[e];
  barras(document.getElementById("e-graf"),linhas.map(r=>({rot:r.grupo,v:r.suprimido?null:r[k+" resíduo médio"],cor:(r[k+" resíduo médio"]??0)>=0?AZUL:LARANJA})),{f:v=>k==="prop"?fmt(100*v,1)+" p.p.":fmt(v,1)+" pts",esq:150});
  document.getElementById("e-tab").innerHTML="<tr><th>Grupo</th><th></th><th>Escolas</th><th>Resíduo médio</th><th>Erro absoluto médio</th></tr>"+linhas.map(r=>r.suprimido?`<tr><td>${r.grupo}</td><td></td><td colspan="3" class="sup">suprimido</td></tr>`:`<tr><td>${r.grupo}</td><td></td><td>${fmt(r.escolas,0)}</td><td>${k==="prop"?fmt(100*r[k+" resíduo médio"],1)+" p.p.":fmt(r[k+" resíduo médio"],1)}</td><td>${k==="prop"?fmt(100*r[k+" erro absoluto médio"],1)+" p.p.":fmt(r[k+" erro absoluto médio"],1)}</td></tr>`).join("");}
["e-desf","e-eixo"].forEach(i=>document.getElementById(i).onchange=erro);
document.getElementById("r-nmin").textContent=D.meta.n_min;
document.getElementById("rodape").textContent=`Fonte: microdados do Saeb 2023 (Inep, CC BY 4.0). Elaboração própria. Versão do código: ${D.meta.versao}.`;
geo();inse();temas();erro();addEventListener("resize",()=>{geo();inse();temas();erro();});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    from .config import PAINEL_N_MIN
    pasta_painel, pasta_tab, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    info = json.loads((pasta_painel.parent / "run_info.json").read_text()) if (pasta_painel.parent / "run_info.json").exists() else {}
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(gerar_html(pasta_painel, pasta_tab, PAINEL_N_MIN, info.get("git", "?")))
    print(f"[ok] {saida}")
