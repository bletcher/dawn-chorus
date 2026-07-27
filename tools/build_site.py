"""
Build a self-contained dawn-chorus dashboard (Observable Plot) from a detection source.

Runs the dawnchorus pipeline, then writes a single HTML file with the data embedded
and interactive charts:
  * Dawn timeline   - onset->offset range per species for a morning, coloured by occupancy
  * Cumulative call distributions (ECDF) - F(t) per species, the robust full-distribution view
  * Occupancy heatmap - species x solar-minute, fraction of mornings singing
  * Species table

Uses the Observable Plot *library* (vendored locally at `site/vendor/plot.umd.min.js`) --
no Observable platform/account, no CDN, no build step, no server. Just open the file, or
host the `site/` folder anywhere.

    python tools/build_site.py --from-analyzer data/results \
        --lat 42.53 --lon -72.53 --tz America/New_York --min-confidence 0.5 \
        --out site/index.html

Re-run whenever new mornings come in; it regenerates the page in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import dawnchorus as dc
from dawnchorus.phenology import DEFAULTS, _anchor_col


def build_data(analyzer_path=None, db_path=None, lat=None, lon=None, tz=None,
               min_conf=0.5, file_tz=None):
    out = dc.run(db_path=db_path, analyzer_path=analyzer_path, latitude=lat, longitude=lon,
                 tz=tz, min_confidence=min_conf, file_tz=file_tz)
    det, ms, ec = out["detections"], out["morning_summary"], out["species_ecdf_month"]

    cfg = DEFAULTS
    acol = _anchor_col(cfg["anchor"])            # min_from_dawn
    lo, hi, bw = cfg["window_start_min"], cfg["window_end_min"], cfg["bin_min"]
    win = det[(det[acol] >= lo) & (det[acol] < hi)].copy()

    mornings = sorted(str(d) for d in pd.unique(ms["date"]))
    n_morn = max(1, win["date"].nunique())

    # Occupancy grid: fraction of mornings a species is present in each 5-min bin.
    edges = np.arange(lo, hi + bw, bw)
    win["bin"] = pd.cut(win[acol], edges, labels=edges[:-1])
    grid = (win.groupby(["common_name", "bin"], observed=True)["date"].nunique()
            .unstack(fill_value=0) / n_morn)

    totals = win.groupby("common_name").size().sort_values(ascending=False)
    heat_species_set = {s for s in totals.index if totals[s] >= 5}
    mean_t = win.groupby("common_name")[acol].mean().sort_values()   # earliest first
    heat_order = [s for s in mean_t.index if s in heat_species_set]

    occ_rows = []
    for sp in heat_order:
        for b, val in grid.loc[sp].items():
            if val > 0:
                occ_rows.append({"name": sp, "t": float(b) + bw / 2.0, "occ": round(float(val), 3)})

    s = ms.rename(columns={"scientific_name": "sci", "common_name": "name",
                           "n_detections": "n", "onset_min": "onset", "offset_min": "offset",
                           "span_min": "span", "peak_min": "peak", "occupancy": "occ",
                           "raw_first_min": "first", "raw_last_min": "last"})
    s["date"] = s["date"].astype(str)
    s = s.round({"onset": 1, "offset": 1, "span": 1, "peak": 1, "occ": 2, "first": 1, "last": 1})
    summary = json.loads(s[["date", "name", "sci", "n", "onset", "offset", "span",
                            "peak", "occ", "first", "last"]].to_json(orient="records"))

    if ec is not None and not ec.empty:
        e = ec.rename(columns={"common_name": "name", "t_min": "t", "F_p25": "lo", "F_p75": "hi"})
        e = e.round({"t": 1, "F": 4, "lo": 4, "hi": 4})
        ecdf = json.loads(e[["name", "t", "F", "lo", "hi"]].to_json(orient="records"))
        ecdf_species = list(dict.fromkeys(e["name"]))
    else:
        ecdf, ecdf_species = [], []

    valid = ms.dropna(subset=["onset_min"])
    earliest = None
    if not valid.empty:
        r = valid.loc[valid["onset_min"].idxmin()]
        earliest = {"name": r["common_name"], "onset": round(float(r["onset_min"]), 1),
                    "date": str(r["date"])}

    meta = {
        "mornings": mornings,
        "n_species": int(det["scientific_name"].nunique()),
        "n_detections": int(len(det)),
        "min_confidence": min_conf,
        "lat": lat, "lon": lon, "tz": tz,
        "anchor": cfg["anchor"], "window": [lo, hi], "bin": bw,
        "earliest": earliest,
        "heat_species": heat_order,
        "ecdf_species": ecdf_species,
        "top_species": list(totals.index),
    }
    return {"meta": meta, "summary": summary, "ecdf": ecdf, "occ": occ_rows}


def render_html(data: dict) -> str:
    return TEMPLATE.replace("/*__DATA__*/", json.dumps(data, allow_nan=False))


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the dawn-chorus dashboard site")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--db")
    src.add_argument("--from-analyzer", dest="from_analyzer")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--file-tz", dest="file_tz", default=None)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--out", default="site/index.html")
    args = p.parse_args(argv)

    data = build_data(analyzer_path=args.from_analyzer, db_path=args.db, lat=args.lat,
                      lon=args.lon, tz=args.tz, min_conf=args.min_confidence, file_tz=args.file_tz)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    m = data["meta"]
    print(f"wrote {out}  ({m['n_detections']:,} detections, {m['n_species']} species, "
          f"{len(m['mornings'])} mornings)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dawn Chorus</title>
<script src="vendor/d3.min.js"></script>
<script src="vendor/plot.umd.min.js"></script>
<style>
  :root{
    --bg:#eaeef4; --surface:#ffffff; --ink:#0f1622; --ink2:#48525f; --muted:#77818f;
    --grid:#e2e7ee; --line:#d6dde6; --accent:#2a78d6; --dawn:#c26a1b;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
    --seq-lo:#e7f0fb; --seq-hi:#12508f;
    --shadow:0 1px 2px rgba(15,22,34,.06), 0 8px 24px rgba(15,22,34,.06);
    color-scheme:light;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    --bg:#0b0f16; --surface:#141a23; --ink:#eef2f8; --ink2:#aab4c2; --muted:#7c8695;
    --grid:#232b37; --line:#2b3542; --accent:#3987e5; --dawn:#e0913f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --seq-lo:#172433; --seq-hi:#7db0ec; --shadow:0 1px 2px rgba(0,0,0,.4);
    color-scheme:dark;
  }}
  :root[data-theme="dark"]{
    --bg:#0b0f16; --surface:#141a23; --ink:#eef2f8; --ink2:#aab4c2; --muted:#7c8695;
    --grid:#232b37; --line:#2b3542; --accent:#3987e5; --dawn:#e0913f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --seq-lo:#172433; --seq-hi:#7db0ec; --shadow:0 1px 2px rgba(0,0,0,.4);
    color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--bg); color:var(--ink);
    font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:1080px; margin:0 auto; padding:28px 20px 64px}
  .display{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;}
  header.masthead{display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
    padding-bottom:18px; border-bottom:1px solid var(--line); margin-bottom:22px;}
  h1{font-size:34px; line-height:1.05; margin:0 0 4px; letter-spacing:-.01em; text-wrap:balance;}
  .sub{color:var(--muted); font-size:13.5px}
  .tag{color:var(--dawn); font-weight:600}
  button.theme{background:var(--surface); color:var(--ink2); border:1px solid var(--line);
    border-radius:8px; padding:7px 11px; font-size:13px; cursor:pointer; white-space:nowrap}
  button.theme:hover{border-color:var(--accent); color:var(--ink)}
  .tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px}
  .tile{background:var(--surface); border:1px solid var(--line); border-radius:10px;
    padding:13px 15px; box-shadow:var(--shadow)}
  .tile .k{font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em}
  .tile .v{font-size:26px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums}
  .tile .v small{font-size:13px; color:var(--ink2); font-weight:500}
  section.card{background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:18px 18px 8px; margin-bottom:20px; box-shadow:var(--shadow)}
  .card h2{font-size:19px; margin:0 0 3px; letter-spacing:-.01em}
  .card .lead{color:var(--ink2); font-size:13.5px; margin:0 0 14px; max-width:62ch}
  .controls{display:flex; flex-wrap:wrap; gap:14px 18px; align-items:center; margin-bottom:12px}
  label.ctl{font-size:13px; color:var(--ink2); display:flex; align-items:center; gap:7px}
  select{font:inherit; font-size:13px; padding:5px 8px; border-radius:7px; border:1px solid var(--line);
    background:var(--surface); color:var(--ink)}
  .chips{display:flex; flex-wrap:wrap; gap:7px}
  .chip{font-size:12.5px; border:1px solid var(--line); border-radius:999px; padding:4px 10px;
    cursor:pointer; color:var(--ink2); user-select:none; background:var(--surface)}
  .chip input{position:absolute; opacity:0; width:0; height:0}
  .chip.on{background:var(--accent); border-color:var(--accent); color:#fff}
  .chip:focus-within{outline:2px solid var(--accent); outline-offset:2px}
  .plot{overflow-x:auto}
  .empty{color:var(--muted); font-size:14px; padding:22px 4px}
  table{border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums}
  thead th{text-align:right; color:var(--muted); font-weight:600; padding:7px 10px;
    border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--surface)}
  thead th:first-child, tbody td:first-child{text-align:left}
  tbody td{padding:6px 10px; border-bottom:1px solid var(--grid)}
  tbody tr:hover{background:color-mix(in srgb, var(--accent) 7%, transparent)}
  .tableScroll{max-height:420px; overflow:auto; border:1px solid var(--grid); border-radius:8px}
  footer{color:var(--muted); font-size:12.5px; margin-top:26px; line-height:1.7}
  footer a{color:var(--accent)}
  figure{margin:0}
  @media (max-width:720px){ .tiles{grid-template-columns:repeat(2,1fr)} h1{font-size:27px} }
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div>
      <h1 class="display">Dawn&nbsp;Chorus</h1>
      <div class="sub" id="subline"></div>
    </div>
    <button class="theme" id="theme" aria-label="Toggle light or dark theme">◐ Theme</button>
  </header>

  <div class="tiles" id="tiles"></div>

  <section class="card">
    <h2 class="display">Who sings when</h2>
    <p class="lead">Each bar spans a species' vocal activity for the chosen morning &mdash; from onset
      (5th percentile of detection times) to offset (95th) &mdash; in minutes from <span class="tag">civil
      dawn</span> (the dashed line). Darker bars are sung more continuously; the tick marks the busiest minute.</p>
    <div class="controls">
      <label class="ctl">Morning <select id="morningSel"></select></label>
    </div>
    <div class="plot" id="chart-timeline"></div>
  </section>

  <section class="card">
    <h2 class="display">Cumulative call distributions</h2>
    <p class="lead">The empirical CDF <em>F(t)</em> &mdash; the share of a species' detections that have
      occurred by each minute. Onset reads off where a curve crosses 0.05, median song-time at 0.5, offset
      at 0.95. Robust by construction: one stray detection shifts a curve by only 1/n.</p>
    <div class="controls"><div class="chips" id="ecdfChips"></div></div>
    <div class="plot" id="chart-ecdf"></div>
  </section>

  <section class="card">
    <h2 class="display">Occupancy across the morning</h2>
    <p class="lead">Species &times; solar-minute. Colour is the fraction of mornings a species was detected
      in each 5-minute bin &mdash; a map of the chorus filling in and thinning out around dawn.</p>
    <div class="plot" id="chart-heat"></div>
  </section>

  <section class="card">
    <h2 class="display">Per-species table</h2>
    <p class="lead">Every species for the chosen morning. Onset/offset/span in minutes from civil dawn;
      occupancy is the share of active bins between onset and offset.</p>
    <div class="tableScroll"><table id="tbl"></table></div>
  </section>

  <footer id="foot"></footer>
</div>

<script type="application/json" id="data">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const {meta, summary, ecdf, occ} = DATA;
const root = document.documentElement;
const css = v => getComputedStyle(root).getPropertyValue(v).trim();
const fmt = (x, d=0) => x==null ? "—" : Number(x).toFixed(d);
const seriesColors = () => ["--s1","--s2","--s3","--s4","--s5","--s6","--s7","--s8"].map(css);

// stable colour per ECDF species (identity, not rank)
const colorFor = {};
function refreshColors(){ const c=seriesColors(); meta.ecdf_species.forEach((s,i)=> colorFor[s]=c[i%8]); }

function plotStyle(){ return {background:"transparent", color:css("--ink"), fontSize:"12.5px"}; }
function W(el){ return Math.max(300, el.clientWidth || 900); }

function renderSubline(){
  const m=meta; const loc = (m.lat!=null&&m.lon!=null) ? `${m.lat.toFixed(2)}, ${m.lon.toFixed(2)}` : "";
  document.getElementById("subline").innerHTML =
    `${m.mornings.length} morning${m.mornings.length>1?"s":""} &middot; ${loc} &middot; ${m.tz||""}`;
}

function renderTiles(){
  const m=meta, e=m.earliest;
  const tiles=[
    ["Mornings", m.mornings.join(" → ")],
    ["Species", m.n_species],
    ["Detections", m.n_detections.toLocaleString()+` <small>&ge;${m.min_confidence} conf</small>`],
    ["Earliest onset", e ? `${fmt(e.onset,0)}<small> min &middot; ${e.name}</small>` : "—"],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(([k,v])=>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

function renderTimeline(){
  const date = document.getElementById("morningSel").value;
  const rows = summary.filter(d=>d.date===date && d.onset!=null).sort((a,b)=>a.onset-b.onset);
  const el = document.getElementById("chart-timeline"); el.innerHTML="";
  if(!rows.length){ el.innerHTML='<p class="empty">No species cleared the onset threshold this morning.</p>'; return; }
  const names = rows.map(d=>d.name);
  el.append(Plot.plot({
    width:W(el), height: names.length*26+80, marginLeft:172, marginRight:26,
    style:plotStyle(),
    x:{label:"minutes from civil dawn →", grid:true, domain:meta.window},
    y:{domain:names, label:null},
    color:{type:"linear", domain:[0,1], range:[css("--seq-lo"),css("--seq-hi")],
           legend:true, label:"occupancy (share of active bins)"},
    marks:[
      Plot.ruleX([0], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
      Plot.barX(rows, {x1:"onset", x2:"offset", y:"name", fill:"occ", rx:3, insetTop:5, insetBottom:5}),
      Plot.tickX(rows, {x:"peak", y:"name", stroke:css("--ink"), strokeWidth:2, strokeOpacity:.85}),
      Plot.dot(rows, {x:"onset", y:"name", r:3.4, fill:css("--ink"), stroke:css("--surface"), strokeWidth:1}),
      Plot.text(rows, {x:"offset", y:"name", text:d=>d.n, dx:8, textAnchor:"start",
                       fill:css("--muted"), fontSize:10}),
      Plot.tip(rows, Plot.pointerY({x:"onset", y:"name",
        title:d=>`${d.name}\nonset ${fmt(d.onset,0)}  offset ${fmt(d.offset,0)} min\nspan ${fmt(d.span,0)} min · occ ${fmt(d.occ,2)}\n${d.n} detections`}))
    ]
  }));
}

function renderEcdf(){
  const sel = [...document.querySelectorAll(".ecdf-sp:checked")].map(c=>c.value);
  const el = document.getElementById("chart-ecdf"); el.innerHTML="";
  const rows = ecdf.filter(d=>sel.includes(d.name));
  if(!rows.length){ el.innerHTML='<p class="empty">Pick one or more species above.</p>'; return; }
  el.append(Plot.plot({
    width:W(el), height:410, marginLeft:52, marginRight:20,
    style:plotStyle(),
    x:{label:"minutes from civil dawn →", grid:true, domain:meta.window},
    y:{label:"↑ cumulative share of detections", domain:[0,1], ticks:[0,.25,.5,.75,1], grid:true},
    color:{domain:sel, range:sel.map(s=>colorFor[s]), legend:true},
    marks:[
      Plot.ruleY([0.05,0.5,0.95], {stroke:css("--grid")}),
      Plot.ruleX([0], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
      Plot.line(rows, {x:"t", y:"F", stroke:"name", strokeWidth:2, curve:"linear", tip:true}),
    ]
  }));
}

function renderHeat(){
  const el = document.getElementById("chart-heat"); el.innerHTML="";
  const names = meta.heat_species;
  if(!names.length){ el.innerHTML='<p class="empty">Not enough detections yet for the heatmap.</p>'; return; }
  const half = meta.bin/2;
  el.append(Plot.plot({
    width:W(el), height: names.length*20+80, marginLeft:172, marginRight:26,
    style:plotStyle(),
    x:{label:"minutes from civil dawn →", domain:meta.window, grid:false},
    y:{domain:names, label:null},
    color:{type:"linear", domain:[0,1], range:[css("--seq-lo"),css("--seq-hi")],
           legend:true, label:"fraction of mornings singing"},
    marks:[
      Plot.rect(occ, {x1:d=>d.t-half, x2:d=>d.t+half, y:"name", fill:"occ", inset:0.5,
        title:d=>`${d.name}\n${fmt(d.t,0)} min · ${Math.round(d.occ*100)}% of mornings`}),
      Plot.ruleX([0], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"})
    ]
  }));
}

function renderTable(){
  const date = document.getElementById("morningSel").value;
  const rows = summary.filter(d=>d.date===date)
    .sort((a,b)=> (a.onset==null)-(b.onset==null) || (a.onset-b.onset) || (b.n-a.n));
  const head = ["Species","n","onset","offset","span","peak","occ"];
  const body = rows.map(d=>`<tr><td>${d.name}</td><td>${d.n}</td><td>${fmt(d.onset,0)}</td>`+
    `<td>${fmt(d.offset,0)}</td><td>${fmt(d.span,0)}</td><td>${fmt(d.peak,0)}</td><td>${fmt(d.occ,2)}</td></tr>`).join("");
  document.getElementById("tbl").innerHTML =
    `<thead><tr>${head.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
}

function renderFoot(){
  document.getElementById("foot").innerHTML =
    `Onset/offset are the 5th/95th percentiles of detection times within the [dawn&minus;2h, dawn+4h] window; `+
    `mornings below the per-species detection floor get no onset. BirdNET does not separate song from call, `+
    `so "span" is vocal-activity span, not song-bout length. Charts by `+
    `<a href="https://observablehq.com/plot/" target="_blank" rel="noopener">Observable&nbsp;Plot</a>. `+
    `Regenerate with <code>tools/build_site.py</code> as new mornings arrive.`;
}

function buildControls(){
  const sel = document.getElementById("morningSel");
  sel.innerHTML = meta.mornings.map(d=>`<option value="${d}">${d}</option>`).join("");
  sel.onchange = ()=>{ renderTimeline(); renderTable(); };

  const chips = document.getElementById("ecdfChips");
  const defaults = new Set(meta.top_species.filter(s=>meta.ecdf_species.includes(s)).slice(0,5));
  const order = meta.top_species.filter(s=>meta.ecdf_species.includes(s));
  chips.innerHTML = order.map(s=>{
    const on = defaults.has(s);
    return `<label class="chip${on?" on":""}"><input class="ecdf-sp" type="checkbox" value="${s}"${on?" checked":""}>${s}</label>`;
  }).join("");
  chips.addEventListener("change", e=>{
    e.target.closest(".chip").classList.toggle("on", e.target.checked);
    renderEcdf();
  });
}

function renderAll(){ refreshColors(); renderTimeline(); renderEcdf(); renderHeat(); renderTable(); }

document.getElementById("theme").onclick = ()=>{
  const cur = root.getAttribute("data-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark":"light");
  root.setAttribute("data-theme", cur==="dark" ? "light":"dark");
  renderAll();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>{ if(!root.hasAttribute("data-theme")) renderAll(); });

let rz; addEventListener("resize", ()=>{ clearTimeout(rz); rz=setTimeout(renderAll, 180); });

renderSubline(); renderTiles(); renderFoot(); buildControls(); renderAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
