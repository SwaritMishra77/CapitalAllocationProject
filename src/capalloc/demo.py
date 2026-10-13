"""Build the self-contained, static GitHub Pages research explorer."""

import argparse
import json
from pathlib import Path

import pandas as pd


def build(panel, predictions, metrics, output):
    source = pd.read_csv(panel, dtype={"cik": str})
    forecast = pd.read_csv(predictions, dtype={"cik": str})
    rows = forecast[forecast.cohort == "text"].merge(
        source[["id", "text", "source_url", "capex_rate"]], on="id", validate="one_to_one")
    fields = ["id", "ticker", "cik", "sector", "fiscal_end", "known_at", "target_known_at",
              "source_accession", "source_url", "actual", "persistence", "numeric_residual_ridge",
              "numeric_plus_transformer", "text", "capex_rate"]
    records = rows[fields].fillna("").to_dict("records")
    scores = json.loads(Path(metrics).read_text())
    payload = json.dumps({"rows": records, "test": scores["hypothesis_test"],
                          "counts": scores["cohorts"]["text"]["counts"],
                          "models": scores["cohorts"]["text"]["models"]}).replace("<", "\\u003c")
    template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Explore point-in-time SEC 10-K forecasts and a matched test of financial text.">
<title>Capital Allocation / Research ledger</title>
<style>
:root{--paper:#f6f1e8;--white:#fffdf8;--ink:#2b2223;--muted:#6f6262;--line:#cbbdb3;--plum:#682f46;--orange:#be673d;--soft:#eadbd2;--serif:Georgia,'Times New Roman',serif;font-family:Arial,Helvetica,sans-serif;color:var(--ink);background:var(--paper)}*{box-sizing:border-box}
body{margin:0}button,select{font:inherit}a{color:var(--plum);text-underline-offset:3px}a:hover{color:var(--orange)}
.topline{height:9px;background:linear-gradient(90deg,var(--plum) 0 64%,var(--orange) 64% 100%)}
.shell{max-width:1450px;margin:auto;padding:0 5vw}header{border-bottom:1px solid var(--line)}header .shell{padding-top:35px;padding-bottom:49px}
.brand{display:flex;justify-content:space-between;gap:25px;align-items:center;font-size:11px;font-weight:bold;letter-spacing:.2em;text-transform:uppercase;color:var(--plum)}
.brand span:last-child{color:var(--muted)}h1,h2,h3,p{margin-top:0}h1{font:normal clamp(46px,6vw,88px)/.98 var(--serif);letter-spacing:-.055em;max-width:1050px;margin:64px 0 23px}h1 em{font-weight:normal;color:var(--plum)}
.dek{font:normal clamp(18px,2vw,24px)/1.48 var(--serif);max-width:760px;color:#493a3b;margin-bottom:28px}
.headerbottom{display:flex;justify-content:space-between;align-items:end;gap:24px}.stamp{border-top:2px solid var(--plum);padding-top:12px;font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
main.shell{display:grid;grid-template-columns:minmax(215px,270px) minmax(0,1fr);gap:clamp(30px,5vw,85px);padding-top:50px;padding-bottom:90px}
aside{border-right:1px solid var(--line);padding-right:30px}.rail{position:sticky;top:25px}.rail-title,.kicker{font-size:11px;font-weight:bold;letter-spacing:.2em;text-transform:uppercase;color:var(--plum)}.rail-title{margin:0 0 25px}
.stat{padding:19px 0;border-top:1px solid var(--line)}.stat strong{font:normal 35px/1 var(--serif);display:block;margin:4px 0 8px}.stat span{font-size:12px;line-height:1.45;color:var(--muted)}
.rail-note{font-size:13px;line-height:1.6;color:var(--muted);padding:16px 0;border-top:2px solid var(--ink)}
.section-number{font:italic 18px var(--serif);color:var(--orange);margin-bottom:10px}h2{font:normal clamp(29px,3vw,46px)/1.12 var(--serif);letter-spacing:-.03em;margin:0 0 14px}.intro{color:var(--muted);line-height:1.65;max-width:760px}
.finder{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:end;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:28px 0 30px;margin:30px 0}
label{font-size:11px;letter-spacing:.13em;text-transform:uppercase;font-weight:bold;color:var(--plum);display:block;margin-bottom:11px}
select{appearance:none;width:100%;max-width:550px;border:1px solid var(--line);background:var(--white);border-radius:0;padding:15px 45px 15px 16px;color:var(--ink);background-image:url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="16" height="12"%3E%3Cpath d="m2 2 6 7 6-7" fill="none" stroke="%23682f46" stroke-width="2"/%3E%3C/svg%3E');background-repeat:no-repeat;background-position:right 16px center}
.finder small{font-size:12px;color:var(--muted);line-height:1.4;text-align:right}
.case{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:35px}.case h3{font:normal clamp(30px,4vw,49px) var(--serif);letter-spacing:-.04em;margin-bottom:8px}.case p{color:var(--muted);line-height:1.5;font-size:14px}
.case .filing-id{font-size:12px;color:var(--plum);letter-spacing:.04em;align-self:end;text-align:right}
.chartbox{background:var(--white);border:1px solid var(--line);padding:clamp(18px,3vw,40px);margin-bottom:32px}.charthead{display:flex;justify-content:space-between;gap:18px;align-items:start}.charthead h3{font:normal 24px var(--serif);margin-bottom:8px}.charthead p{font-size:13px;line-height:1.5;color:var(--muted);max-width:550px}.legend{font-size:12px;white-space:nowrap;color:var(--muted)}.swatch{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px;background:var(--orange)}
#plot{display:block;width:100%;height:auto;overflow:visible}.chart-foot{border-top:1px solid var(--line);padding-top:15px;font-size:12px;line-height:1.55;color:var(--muted)}
.reading{display:grid;grid-template-columns:minmax(0,1fr) 155px;gap:30px;border-top:2px solid var(--ink);padding-top:23px}.reading h3{font:normal 27px var(--serif)}.reading p{font-size:13px;line-height:1.6;color:var(--muted)}blockquote{border-left:3px solid var(--orange);padding:2px 0 2px 19px;margin:22px 0 10px;font:normal 17px/1.62 var(--serif);max-height:250px;overflow:auto;white-space:pre-wrap}.source{font-size:12px;line-height:1.7;word-break:break-word}
.verdict{margin-top:55px;background:var(--soft);border-left:5px solid var(--plum);padding:24px 26px}.verdict strong{font:normal 24px var(--serif)}.verdict p{font-size:14px;line-height:1.6;margin:10px 0 0}
footer{border-top:1px solid var(--line);padding:24px 0 35px;font-size:12px;color:var(--muted);line-height:1.6}
@media(max-width:840px){main.shell{display:block;padding-top:32px}aside{border:0;padding:0}.rail{position:static;display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.rail-title,.rail-note{grid-column:1/-1}.stat{border-top:1px solid var(--line)}.rail-note{margin-bottom:30px}.headerbottom{display:block}.stamp{white-space:normal}}@media(max-width:600px){header .shell{padding-top:24px}.brand span:last-child{display:none}h1{margin:43px 0 20px}.finder,.case,.reading{display:block}.finder small,.case .filing-id{text-align:left;display:block;margin-top:12px}.charthead{display:block}.legend{margin:8px 0 15px}.reading .source{margin-top:20px}.rail{gap:8px}.stat strong{font-size:28px}}
</style></head><body><div class="topline"></div><header><div class="shell"><div class="brand"><span>Capital Allocation / Research ledger</span><span>SEC 10-K · 100-company universe</span></div>
<h1>What companies <em>say.</em><br>What they <em>spend.</em></h1><p class="dek">A filing-date study of whether capital spending language helps forecast next year's capital expenditure intensity.</p>
<div class="headerbottom"><span class="kicker">Explore the held-out filings below ↓</span><span class="stamp">Historical test · source documents linked</span></div></div></header>
<main class="shell"><aside><div class="rail"><div class="rail-title">Study at a glance</div><div class="stat"><strong id="n"></strong><span>Matched test filings</span></div><div class="stat"><strong id="companies"></strong><span>Companies with test text</span></div><div class="stat"><strong id="effect"></strong><span>Mean MAE reduction from text, percentage points</span></div><div class="rail-note">The interval for this gain includes zero. These forecasts are a research comparison, not an investment recommendation. <a href="RESULTS.md">Read full results</a>.</div></div></aside>
<article><div class="section-number">01 / The filing ledger</div><h2>Inspect a forecast</h2><p class="intro">Select a company's original 10-K. The graphic compares its following year's reported spending with three forecasts available when that 10-K was filed. All marks use the following year's revenue as the denominator.</p>
<div class="finder"><div><label for="filing">Source company / fiscal year</label><select id="filing"></select></div><small>Filing dates determine<br>the research split.</small></div>
<div class="case"><div><div class="kicker">Selected observation</div><h3 id="name"></h3><p id="meta"></p></div><div class="filing-id" id="accession"></div></div>
<div class="chartbox"><div class="charthead"><div><h3>Capital spending as % of revenue</h3><p>Four points share a common scale. The orange mark is the realized outcome; the other marks are forecasts made from the source filing.</p></div><div class="legend"><span class="swatch"></span>Actual next year</div></div><svg id="plot" viewBox="0 0 720 300" role="img" aria-label="Capital expenditure intensity comparison"></svg><div class="chart-foot">The chart presents a single historical case, not model-wide accuracy. Forecasts may lie outside the visible nonnegative scale; any such value is labeled. Read the aggregate paired comparison below.</div></div>
<div class="reading"><div><div class="section-number">02 / Original disclosure</div><h3>Passage from the source 10-K</h3><p>A rule selects investment-related passages. The excerpt is evidence to inspect and may include boilerplate; it is not a claim about management's intent.</p><blockquote id="excerpt"></blockquote></div><div class="source"><div class="kicker">Primary source</div><p id="dates"></p><a id="sec" target="_blank" rel="noopener noreferrer">Open SEC filing ↗</a></div></div>
<div class="verdict"><div class="kicker">The matched test</div><strong id="verdict-title"></strong><p id="verdict-body"></p></div></article></main>
<footer><div class="shell">Reproducible source, row-level predictions and methods are in this repository. The 24-company cached text cohort is selective, and prior test results were inspected before the availability audit. <a href="METHOD.md">Methods</a> · <a href="AUDIT.md">Audit</a></div></footer>
<script>
const DATA=__PAYLOAD__;const $=id=>document.getElementById(id);const pp=x=>(x*100).toFixed(2)+'%';const svgNS='http://www.w3.org/2000/svg';
const rows=DATA.rows.sort((a,b)=>a.ticker.localeCompare(b.ticker)||a.fiscal_end.localeCompare(b.fiscal_end));
$('n').textContent=DATA.counts.test;$('companies').textContent=DATA.counts.test_companies;$('effect').textContent=(DATA.test.improvement_pp>=0?'+':'')+DATA.test.improvement_pp.toFixed(3);
for(let i=0;i<rows.length;i++){const o=document.createElement('option');o.value=i;o.textContent=rows[i].ticker+' / FY '+rows[i].fiscal_end.slice(0,4);$('filing').append(o)}
const ci=DATA.test.cluster_bootstrap_95_ci_pp;$('verdict-title').textContent=ci[0]<=0&&ci[1]>=0?'No reliable transformer gain in this sample':'Observed difference on matched rows';
$('verdict-body').textContent='On '+DATA.counts.test+' matched filings, numbers alone had '+DATA.models.numeric_residual_ridge.mae_pp.toFixed(3)+' pp MAE versus '+DATA.models.numeric_plus_transformer.mae_pp.toFixed(3)+' pp with frozen FinBERT. Paired improvement: '+DATA.test.improvement_pp.toFixed(3)+' pp; 95% company-cluster interval ['+ci.map(x=>x.toFixed(3)).join(', ')+'] pp. Exploratory result; earlier test results were inspected.';
function svgEl(tag,attrs,content){const node=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(content!==undefined)node.textContent=content;return node}
function draw(r){const values=[['Reported next year',r.actual,'#be673d'],['Last-year rate',r.persistence,'#776d66'],['Numbers only',r.numeric_residual_ridge,'#682f46'],['Numbers + text',r.numeric_plus_transformer,'#9a7280']];
 const upper=Math.max(.015,...values.map(x=>Math.max(0,x[1])))*1.16;const svg=$('plot');svg.replaceChildren();const x0=192,x1=655,y0=28,y1=260;
 for(let i=0;i<=4;i++){const x=x0+(x1-x0)*i/4;svg.append(svgEl('line',{x1:x,y1:y0,x2:x,y2:y1,stroke:'#e4d9d1','stroke-width':1}));svg.append(svgEl('text',{x,y:281,'text-anchor':'middle',fill:'#6f6262','font-size':12,'font-family':'Arial'},pp(upper*i/4)))}
 for(let i=0;i<values.length;i++){const [label,value,color]=values[i];const y=54+i*53;svg.append(svgEl('text',{x:0,y:y+4,fill:'#2b2223','font-size':14,'font-family':'Arial'},label));const x=x0+(x1-x0)*Math.min(1,Math.max(0,value/upper));svg.append(svgEl('line',{x1:x0,y1:y,x2:x,y2:y,stroke:color,'stroke-width':2}));svg.append(svgEl('circle',{cx:x,cy:y,r:i===0?8:6,fill:color,stroke:'#fffdf8','stroke-width':2}));svg.append(svgEl('text',{x:Math.min(x+14,690),y:y+4,fill:color,'font-size':13,'font-weight':'bold','font-family':'Arial'},pp(value)))}
 svg.setAttribute('aria-label','Reported next year '+pp(r.actual)+'; last year '+pp(r.persistence)+'; numbers only '+pp(r.numeric_residual_ridge)+'; numbers and text '+pp(r.numeric_plus_transformer))}
function render(){const r=rows[Number($('filing').value)];$('name').textContent=r.ticker+' / '+r.fiscal_end.slice(0,4);$('meta').textContent=r.sector+' · source 10-K filed '+r.known_at;$('accession').textContent='Source accession '+r.source_accession;$('dates').textContent='Source filed '+r.known_at+'. Following report filed '+r.target_known_at+'.';$('excerpt').textContent=r.text.slice(0,1250)+(r.text.length>1250?'…':'');$('sec').href=r.source_url;draw(r)}
$('filing').addEventListener('change',render);render();
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PAYLOAD__", payload))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--predictions", type=Path, default=Path("reports/test_predictions.csv"))
    p.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    p.add_argument("--out", type=Path, default=Path("docs/index.html"))
    a = p.parse_args()
    build(a.panel, a.predictions, a.metrics, a.out)


if __name__ == "__main__":
    main()
