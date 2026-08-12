"""
Build a self-contained dawn-chorus dashboard (Observable Plot) from a detection source.

Runs the dawnchorus pipeline, then writes a single HTML file with the data embedded and
interactive charts, all governed by one global **time scope** (Aggregate: day/week/month/
year + a period slider). Move the slider and every chart recomputes for that set of mornings:
  * Dawn timeline   - median onset->offset per species, coloured by median occupancy
  * Cumulative call distributions (ECDF) - mean of the period's per-morning curves
  * Occupancy heatmap - species x solar-minute, fraction of the period's mornings singing
  * Species table   - per-species aggregates over the period

The page holds only per-morning building blocks (per-species-per-bin detection counts and a
per-morning onset/offset summary); the browser aggregates them up to the chosen grain, so the
same file explores day-to-day detail or seasonal roll-ups without regenerating.

Uses the Observable Plot *library* (vendored locally at `site/vendor/`) -- no Observable
platform/account, no CDN, no build step, no server. Just open the file, or host `site/`.

    python tools/build_site.py --from-analyzer data/results \
        --lat 42.53 --lon -72.53 --tz America/New_York --min-confidence 0.5 \
        --out site/index.html
"""
from __future__ import annotations

import argparse
import json
import warnings
import wave
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import dawnchorus as dc
from dawnchorus import recorders as rec
from dawnchorus.phenology import DEFAULTS, _anchor_col


def scan_audio(audio_dir, lat, lon, tz, recorder=None):
    """Recordings -> {date: [{name, start-second-of-day, duration}]} plus the civil-dawn
    second-of-day per date, so the page can turn a solar-minute back into file + offset.

    Filenames are parsed through the recorder's convention (sniffed when not named). This
    used to hardcode the Song Meter pattern and silently skip anything else, which meant a
    recorder like the Owl Sense produced charts with no click-to-listen and no explanation.
    """
    solar = dc.SolarModel(lat, lon, tz)
    # One glob, filtered by suffix: globbing "*.wav" AND "*.WAV" double-counts every file
    # on Windows, where the pattern match is already case-insensitive.
    paths = sorted(p for p in Path(audio_dir).glob("*") if p.suffix.lower() == ".wav")
    names = [p.name for p in paths]
    prof = rec.get(recorder)
    conv = None
    if prof is not None:
        conv = rec.CONVENTIONS_BY_ID.get(prof.resolve(names).convention or "")
    conv = conv or rec.sniff(names)

    files, skipped = {}, []
    for p in paths:
        t = conv.parse(p.name) if conv else None
        if t is None:
            skipped.append(p.name)
            continue
        d = t.strftime("%Y-%m-%d")
        start = t.hour * 3600 + t.minute * 60 + t.second
        try:
            with wave.open(str(p)) as w:
                dur = round(w.getnframes() / w.getframerate(), 1)
        except Exception:
            dur = None
        files.setdefault(d, []).append({"name": p.name, "s": start, "d": dur})
    if skipped:
        warnings.warn(
            f"{len(skipped)} recording(s) in {audio_dir} had no parseable timestamp "
            f"(e.g. {skipped[0]}) and will have no click-to-listen; pass recorder= "
            "or add a convention to dawnchorus.recorders.", stacklevel=2)

    dawn = {}
    for d in files:
        y, mo, da = (int(x) for x in d.split("-"))
        dw = solar.dawn(date(y, mo, da))
        if dw is not None:
            dawn[d] = dw.hour * 3600 + dw.minute * 60 + dw.second
        files[d].sort(key=lambda f: f["s"])
    return files, dawn


def build_data(analyzer_path=None, db_path=None, lat=None, lon=None, tz=None,
               min_conf=0.5, file_tz=None, audio_dir=None, audio_base="../data", label_min_conf=0.25,
               label_analyzer_path=None, recorder=None):
    out = dc.run(db_path=db_path, analyzer_path=analyzer_path, latitude=lat, longitude=lon,
                 tz=tz, min_confidence=min_conf, file_tz=file_tz, recorder=recorder)
    det, ms = out["detections"], out["morning_summary"]

    cfg = DEFAULTS
    acol = _anchor_col(cfg["anchor"])            # min_from_dawn
    lo, hi, bw = cfg["window_start_min"], cfg["window_end_min"], cfg["bin_min"]
    win = det[(det[acol] >= lo) & (det[acol] < hi)].copy()

    edges = np.arange(lo, hi + bw, bw)
    nbins = len(edges) - 1
    grid = [round(float(edges[i]) + bw / 2.0, 1) for i in range(nbins)]   # bin centres
    win["bin"] = pd.cut(win[acol], edges, labels=False)                   # 0..nbins-1

    totals = win.groupby("common_name").size().sort_values(ascending=False)
    heat_set = {s for s in totals.index if totals[s] >= 5}
    mean_t = win.groupby("common_name")[acol].mean().sort_values()        # earliest first
    species = [s for s in mean_t.index if s in heat_set]
    sp_idx = {s: i for i, s in enumerate(species)}

    # Per-morning, per-species, per-bin detection COUNTS -- the one building block the page
    # aggregates up to any grain (occupancy, mean-of-morning ECDF, quantile onset/offset).
    hs = win[win["common_name"].isin(heat_set)].dropna(subset=["bin"]).copy()
    hs["bin"] = hs["bin"].astype(int)
    days = sorted(str(d) for d in hs["date"].unique())
    day_idx = {d: i for i, d in enumerate(days)}
    cnt = hs.groupby(["date", "common_name", "bin"]).size()
    counts = [[day_idx[str(d)], sp_idx[s], int(b), int(c)] for (d, s, b), c in cnt.items()]

    day_keys = {}
    for d in days:
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        day_keys[d] = {"year": f"{ts.year}", "month": f"{ts.year}-{ts.month:02d}",
                       "week": f"{int(iso[0])}-W{int(iso[1]):02d}", "day": d}

    s = ms.rename(columns={"scientific_name": "sci", "common_name": "name",
                           "n_detections": "n", "onset_min": "onset", "offset_min": "offset",
                           "span_min": "span", "peak_min": "peak", "occupancy": "occ"})
    s["date"] = s["date"].astype(str)
    s = s.round({"onset": 1, "offset": 1, "span": 1, "peak": 1, "occ": 2})
    summary = json.loads(s[["date", "name", "sci", "n", "onset", "offset", "span",
                            "peak", "occ"]].to_json(orient="records"))

    # Top few detections per (date, species) by confidence -> the audiogram lands on a real
    # call. Each entry is [solar-minute, confidence].
    clips = {}
    if "confidence" in win.columns:
        for (d, sp), g in win.groupby(["date", "common_name"]):
            top = g.nlargest(3, "confidence")
            clips.setdefault(str(d), {})[sp] = [[round(float(r[acol]), 2), round(float(r["confidence"]), 2)]
                                                for _, r in top.iterrows()]

    # Every in-window detection per morning, for labelling the spectrogram:
    # [solar-minute, species-index, confidence]. Reloaded down to `label_min_conf` so the
    # spectrogram shows more calls than the (higher) analysis threshold used for the charts.
    label_src = label_analyzer_path or analyzer_path      # labels can come from a denser (higher-overlap) run
    if label_src:
        det_lab = dc.load_birdnet_analyzer(label_src, min_confidence=label_min_conf,
                                           latitude=lat, longitude=lon, tz=tz, file_tz=file_tz,
                                           recorder=recorder)
    else:
        det_lab = dc.load_detections(db_path, min_confidence=label_min_conf, latitude=lat,
                                     longitude=lon, recorder=recorder)
    det_lab = dc.SolarModel(lat, lon, tz).annotate(det_lab)
    winlab = det_lab[(det_lab[acol] >= lo) & (det_lab[acol] < hi)]
    label_species = sorted(winlab["common_name"].unique().tolist())
    lsp = {s: i for i, s in enumerate(label_species)}
    dets_by_day = {}
    for d, g in winlab.groupby("date"):
        dets_by_day[str(d)] = [[round(float(a), 2), lsp[s], round(float(c), 2)]
                               for s, a, c in zip(g["common_name"], g[acol], g["confidence"])]

    valid = ms.dropna(subset=["onset_min"])
    earliest = None
    if not valid.empty:
        r = valid.loc[valid["onset_min"].idxmin()]
        earliest = {"name": r["common_name"], "onset": round(float(r["onset_min"]), 1)}

    meta = {
        "mornings": sorted(str(d) for d in pd.unique(ms["date"])),
        "days": days,
        "n_species": int(det["scientific_name"].nunique()),
        "n_detections": int(len(det)),
        "min_confidence": min_conf,
        "lat": lat, "lon": lon, "tz": tz,
        "window": [lo, hi], "bin": bw, "grid": grid,
        "species": species,                                   # heatmap/ECDF order (earliest first)
        "top_species": [s for s in totals.index if s in heat_set],   # by detections (ECDF default + colour)
        "earliest": earliest,
        "label_species": label_species,
        "audio_base": audio_base,
        "recorders": (sorted(str(r) for r in det["recorder"].dropna().unique())
                      if "recorder" in det.columns else []),
    }
    audio, dawn = scan_audio(audio_dir, lat, lon, tz, recorder) if audio_dir else ({}, {})
    # Civil dawn per morning, independent of whether recordings are present: the viewer
    # needs it to offer a clock-time axis, and dawn drifts through the season so a fixed
    # offset would misplace every point.
    solar = dc.SolarModel(lat, lon, tz)
    for d in meta["mornings"]:
        if d in dawn:
            continue
        y, mo, da = (int(x) for x in d.split("-"))
        dw = solar.dawn(date(y, mo, da))
        if dw is not None:
            dawn[d] = dw.hour * 3600 + dw.minute * 60 + dw.second
    return {"meta": meta, "summary": summary, "counts": counts, "day_keys": day_keys,
            "audio": audio, "dawn": dawn, "clips": clips, "dets": dets_by_day}


def render_html(data: dict) -> str:
    return (TEMPLATE.replace("<!--__VIEWER_BOOTSTRAP__-->", "")
                    .replace("/*__DATA__*/", json.dumps(data, allow_nan=False)))


# Injected into viewer.html in place of embedded data: fetch the site list + the selected
# site's payload from the API, add a site picker to the masthead, then boot() on it.
# RAW string: this is JavaScript, and its regexes contain backslash escapes that Python
# would otherwise try (and fail) to interpret.
VIEWER_BOOTSTRAP = r"""<script>
window.__bootFetch = async function(boot){
  const RAW = %BASE%, MODE = %MODE%;                         // static JSON files, or a live API

  // Resolve a RELATIVE data base against the document's DIRECTORY, not against whatever
  // the browser guesses. Visiting the site without a trailing slash (/dawn-chorus rather
  // than /dawn-chorus/) makes "./data/sites.json" resolve to /data/sites.json, one level
  // too high, and every fetch 404s -- the page loads but shows no data at all.
  const docDir = (function(){
    const p = location.pathname;
    if (p.endsWith("/")) return p;
    const last = p.slice(p.lastIndexOf("/") + 1);
    // A final segment with a dot is a file (index.html); without one it's a directory
    // the server did not redirect to its canonical trailing-slash form.
    return last.includes(".") ? p.slice(0, p.lastIndexOf("/") + 1) : p + "/";
  })();
  const isAbsolute = /^(https?:)?\/\//.test(RAW) || RAW.startsWith("/");
  const BASE = (MODE === "api" || isAbsolute) ? RAW
    : new URL(RAW.replace(/^\.\//, ""), location.origin + docDir).href.replace(/\/+$/, "");

  const listUrl = MODE === "api" ? BASE + "/sites" : BASE + "/sites.json";
  const dataUrl = s => MODE === "api" ? BASE + "/sites/" + encodeURIComponent(s) + "/data"
                                      : BASE + "/" + encodeURIComponent(s) + ".json";
  const sub = document.getElementById("subline");
  let sites;
  try {
    const r = await fetch(listUrl);
    if (!r.ok) throw new Error(r.status);
    sites = await r.json();
  }
  catch (e) { sub.textContent = "Can't load the site list (" + listUrl + ")"; return; }
  const slug = new URLSearchParams(location.search).get("site") || (sites[0] && sites[0].slug);
  const sel = document.createElement("select");
  sel.id = "siteSel"; sel.title = "Site";
  sites.forEach(s => {
    const o = document.createElement("option");
    o.value = s.slug; o.textContent = s.name; o.selected = (s.slug === slug);
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => {
    const p = new URLSearchParams(location.search); p.set("site", sel.value);
    location.search = p.toString();                         // reload -> single clean boot
  });
  const hb = document.querySelector(".hbtns");
  if (hb) hb.insertBefore(sel, hb.firstChild);
  if (!slug) { sub.textContent = "No sites yet."; return; }
  let data;
  try { data = await (await fetch(dataUrl(slug))).json(); }
  catch (e) { sub.textContent = "No data for " + slug; return; }
  boot(data);
};
</script>"""


def render_viewer(base: str, mode: str) -> str:
    """mode 'static' -> fetch <base>/sites.json + <base>/<slug>.json (files on S3/CloudFront);
    mode 'api' -> fetch <base>/sites + <base>/sites/<slug>/data (the FastAPI server)."""
    boot = (VIEWER_BOOTSTRAP.replace("%BASE%", json.dumps(base.rstrip("/")))
                            .replace("%MODE%", json.dumps(mode)))
    return (TEMPLATE.replace("<!--__VIEWER_BOOTSTRAP__-->", boot)
                    .replace("/*__DATA__*/", ""))          # empty #data -> __bootFetch runs


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the dawn-chorus dashboard site")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--db")
    src.add_argument("--from-analyzer", dest="from_analyzer")
    p.add_argument("--label-from-analyzer", dest="label_from_analyzer", default=None,
                   help="separate (denser, e.g. higher-overlap) results folder for spectrogram labels only")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--recorder", default=None,
                   help="recorder profile id (see dawnchorus/recorders.py); supplies the filename "
                        "convention + clock zone and tags the detections")
    p.add_argument("--file-tz", dest="file_tz", default=None)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--label-min-confidence", dest="label_min_conf", type=float, default=0.25,
                   help="confidence floor for spectrogram labels (charts use --min-confidence)")
    p.add_argument("--audio", default=None,
                   help="folder of WAV recordings (enables click-to-listen in daily scope)")
    p.add_argument("--audio-url-base", dest="audio_base", default="../data",
                   help="URL prefix the page uses to reach recordings (default ../data)")
    p.add_argument("--out", default="site/dashboard-local.html",
                   help="local click-to-listen dashboard (git-ignored); the public site is the viewer "
                        "from build_viewer.py")
    args = p.parse_args(argv)

    data = build_data(analyzer_path=args.from_analyzer, db_path=args.db, lat=args.lat,
                      lon=args.lon, tz=args.tz, min_conf=args.min_confidence, file_tz=args.file_tz,
                      audio_dir=args.audio, audio_base=args.audio_base, label_min_conf=args.label_min_conf,
                      label_analyzer_path=args.label_from_analyzer, recorder=args.recorder)
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
    --spine-a:#7c93b8; --spine-b:#e8c79a;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
    --seq-lo:#e7f0fb; --seq-hi:#12508f;
    --scope:#e6eef9;
    --shadow:0 1px 2px rgba(15,22,34,.06), 0 8px 24px rgba(15,22,34,.06);
    color-scheme:light;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    --bg:#0b0f16; --surface:#141a23; --ink:#eef2f8; --ink2:#aab4c2; --muted:#7c8695;
    --grid:#232b37; --line:#2b3542; --accent:#3987e5; --dawn:#e0913f;
    --spine-a:#243c63; --spine-b:#8a5a24;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --seq-lo:#172433; --seq-hi:#7db0ec; --scope:#161f2e; --shadow:0 1px 2px rgba(0,0,0,.4);
    color-scheme:dark;
  }}
  :root[data-theme="dark"]{
    --bg:#0b0f16; --surface:#141a23; --ink:#eef2f8; --ink2:#aab4c2; --muted:#7c8695;
    --grid:#232b37; --line:#2b3542; --accent:#3987e5; --dawn:#e0913f;
    --spine-a:#243c63; --spine-b:#8a5a24;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --seq-lo:#172433; --seq-hi:#7db0ec; --scope:#161f2e; --shadow:0 1px 2px rgba(0,0,0,.4);
    color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--bg); color:var(--ink);
    font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; -webkit-font-smoothing:antialiased;}
  /* Fill the screen, but not without limit: chart heights scale with width (see ratioH)
     so a wide window gets BIGGER charts rather than flatter ones. The cap keeps an
     ultrawide from turning the prose and the scope bar into a single unreadable line. */
  .wrap{max-width:min(2100px, 96vw); margin:0 auto; padding:28px 24px 64px}
  .display{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;}
  header.masthead{display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
    padding-bottom:18px; border-bottom:1px solid var(--line); margin-bottom:22px;}
  h1{font-size:34px; line-height:1.05; margin:0 0 4px; letter-spacing:-.01em; text-wrap:balance;}
  .sub{color:var(--muted); font-size:13.5px}
  .tag{color:var(--dawn); font-weight:600}
  button.theme{background:var(--surface); color:var(--ink2); border:1px solid var(--line);
    border-radius:8px; padding:7px 11px; font-size:13px; cursor:pointer; white-space:nowrap}
  button.theme:hover{border-color:var(--accent); color:var(--ink)}
  .hbtns{display:flex; gap:8px}
  /* Collapsible cards. The heading is the control, so every card behaves the same way
     and nothing but the heading survives collapsing. */
  .card > h2{cursor:pointer; user-select:none; display:flex; align-items:center; gap:9px}
  .card > h2::before{content:"▾"; font-size:.72em; color:var(--muted); transition:transform .15s;
    display:inline-block; width:.8em}
  .card.collapsed > h2::before{transform:rotate(-90deg)}
  .card.collapsed > *:not(h2){display:none}
  .card > h2 .why{margin-left:auto; font-size:11px; font-weight:400; color:var(--muted);
    letter-spacing:.02em}
  .setrow{display:flex; flex-direction:column; gap:6px; margin:0 0 18px}
  .setrow > b{font-size:13px}
  .setrow label{display:flex; gap:8px; align-items:flex-start; font-size:13.5px; cursor:pointer}
  .setrow small{color:var(--muted); display:block; font-size:12px; line-height:1.45}
  .modal{position:fixed; inset:0; z-index:50; display:flex; align-items:flex-start; justify-content:center;
    padding:44px 16px; background:rgba(9,13,20,.55); overflow:auto}
  .modal[hidden]{display:none}
  .dialog{position:relative; width:100%; max-width:680px; background:var(--surface); color:var(--ink);
    border:1px solid var(--line); border-radius:14px; padding:26px 30px 30px; box-shadow:0 20px 60px rgba(0,0,0,.4)}
  .dialog h2{font-size:24px; margin:0 0 12px}
  .dialog h3{font-size:15px; margin:20px 0 5px; letter-spacing:-.01em}
  .dialog p, .dialog li{font-size:13.5px; color:var(--ink2); line-height:1.6}
  .dialog p{margin:0 0 8px} .dialog ul{margin:0 0 8px; padding-left:20px} .dialog li{margin:2px 0}
  .dialog b, .dialog strong{color:var(--ink); font-weight:600}
  .dialog .tag{color:var(--dawn)} .dialog .muted{color:var(--muted); font-weight:400}
  .dialog ol{margin:0 0 10px; padding-left:20px} .dialog ol li{margin:6px 0}
  .dialog pre{background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:10px 12px;
    overflow-x:auto; font-size:12px; line-height:1.5; white-space:pre; margin:6px 0; color:var(--ink)}
  .copybtn{font:inherit; font-size:11px; padding:3px 10px; border-radius:6px; border:1px solid var(--line);
    background:var(--surface); color:var(--ink2); cursor:pointer; margin:0 0 10px}
  .copybtn:hover{border-color:var(--accent); color:var(--ink)}
  .close{position:absolute; top:10px; right:14px; background:none; border:none; color:var(--muted);
    font-size:26px; line-height:1; cursor:pointer; padding:2px 6px}
  .close:hover{color:var(--ink)}
  .tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px}
  .tile{background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:13px 15px; box-shadow:var(--shadow)}
  .tile .k{font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em}
  .tile .v{font-size:26px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums}
  .tile .v small{font-size:13px; color:var(--ink2); font-weight:500}
  /* Column of rows: each control group gets its own line with a matching eyebrow label,
     so "Time axis" reads as a peer of "Time scope" rather than one more inline control. */
  .scopebar{position:sticky; top:0; z-index:20; display:flex; flex-direction:column;
    align-items:stretch; gap:10px;
    background:var(--scope); border:1px solid var(--line); border-radius:11px;
    padding:12px 16px; margin-bottom:22px; box-shadow:var(--shadow); backdrop-filter:saturate(1.2)}
  .scoperow{display:flex; align-items:center; gap:16px 20px; flex-wrap:wrap}
  .scopebar .eyebrow{font-size:11px; text-transform:uppercase; letter-spacing:.09em;
    color:var(--muted); font-weight:600; min-width:78px}
  .periodlabel{font-size:14px; color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums; white-space:nowrap}
  .scopehint{margin-left:auto; font-size:12px; color:var(--muted)}
  label.ctl{font-size:13px; color:var(--ink2); display:flex; align-items:center; gap:7px}
  label.ctl.grow{flex:1; min-width:220px}
  select{font:inherit; font-size:13px; padding:5px 8px; border-radius:7px; border:1px solid var(--line); background:var(--surface); color:var(--ink)}
  /* Segmented control: two mutually exclusive states, so it reads as one control with a
     current position rather than two independent buttons. */
  .seg{display:inline-flex; border:1px solid var(--line); border-radius:7px; overflow:hidden;
    background:var(--surface)}
  .seg button{font:inherit; font-size:12.5px; padding:5px 10px; border:0; cursor:pointer;
    background:transparent; color:var(--ink2); white-space:nowrap}
  .seg button + button{border-left:1px solid var(--line)}
  .seg button:hover:not([aria-pressed="true"]):not(:disabled){background:var(--scope); color:var(--ink)}
  .seg button[aria-pressed="true"]{background:var(--accent); color:#fff; font-weight:600}
  .seg button:disabled{opacity:.4; cursor:not-allowed}
  input[type=range]{flex:1; accent-color:var(--accent); cursor:pointer; min-width:140px}
  input[type=range]:disabled{opacity:.45; cursor:default}
  /* ── Chapters ─────────────────────────────────────────────────────────────────────
     The page is a morning. Each chapter states a solar moment, the question it answers
     and what the data says, then shows the charts that answer it. The rail sticks so the
     question stays on screen while you read the chart -- the thing that makes this a
     narrative rather than a stack of plots. */
  .chapters{position:relative}
  .chapter{display:grid; grid-template-columns:230px minmax(0,1fr); gap:26px;
    align-items:start; margin:0 0 40px}
  .rail{position:sticky; top:112px; padding-left:20px}
  /* The spine is the signature: a vertical gradient from pre-dawn blue through the dawn
     amber to daylight, with a node at each chapter. It is not decoration -- it is the
     same solar axis every chart on the page is plotted against. */
  .rail::before{content:""; position:absolute; left:0; top:6px; bottom:-40px; width:2px;
    background:linear-gradient(180deg, var(--spine-a), var(--spine-b))}
  .chapter:last-child .rail::before{bottom:auto; height:64px}
  .rail::after{content:""; position:absolute; left:-4px; top:6px; width:10px; height:10px;
    border-radius:50%; background:var(--dawn); box-shadow:0 0 0 3px var(--bg)}
  .chwhen{display:block; font-family:var(--mono,ui-monospace,SFMono-Regular,Menlo,monospace);
    font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--dawn);
    font-weight:700; margin-bottom:7px}
  .chtitle{font-size:20px; line-height:1.15; margin:0 0 8px; letter-spacing:-.01em;
    text-wrap:balance}
  .chq{font-size:13.5px; color:var(--ink2); margin:0 0 10px; line-height:1.5}
  .chfind{font-size:13px; color:var(--muted); margin:0; line-height:1.55;
    border-left:2px solid var(--line); padding-left:10px}
  .chfind:empty{display:none}
  .chbody{min-width:0}
  /* Two charts that are meant to be compared sit side by side, not 400px apart. */
  .chbody.pair{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px}
  .chbody.pair > .card{margin-bottom:0}
  @media (max-width:1100px){ .chbody.pair{grid-template-columns:1fr} .chbody.pair > .card{margin-bottom:20px} }
  @media (max-width:900px){
    /* Rail becomes a header: the question still arrives before the chart, which is the
       point of the narrative -- it just stops being a column. */
    .chapter{grid-template-columns:1fr; gap:12px; margin-bottom:30px}
    .rail{position:static; padding-left:16px}
    .rail::before{top:2px; bottom:2px}
    .rail::after{top:2px}
    .chtitle{font-size:18px}
  }
  /* display:contents by default so the scope bar stays a child of .wrap and its
     sticky containing block remains the whole page -- wrapping it in a grid would
     confine the pinning to one row. Only when Listen is actually on screen, and the
     window is wide enough for both, does this become a real two-column row. */
  .topline{display:contents}
  @media (min-width:1180px){
    .topline:has(#audioCard:not([hidden])){display:grid; align-items:start;
      grid-template-columns:minmax(0,1fr) minmax(400px,44%); gap:18px; margin-bottom:22px}
    .topline:has(#audioCard:not([hidden])) > .scopebar{margin-bottom:0; align-self:start}
    .topline:has(#audioCard:not([hidden])) > .card.audio{margin-bottom:0}
    /* Pinned together while you are listening: change scope or species and the player
       stays put. Shorter spectrogram so the pair does not eat the viewport. */
    .topline:has(#audioCard:not([hidden])) > .card.audio{position:sticky; top:0}
    .topline:has(#audioCard:not([hidden])) #spec{height:150px}
  }
  section.card{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px 18px 8px; margin-bottom:20px; box-shadow:var(--shadow)}
  .card h2{font-size:19px; margin:0 0 3px; letter-spacing:-.01em}
  .card .lead{color:var(--ink2); font-size:13.5px; margin:0 0 14px; max-width:64ch}
  .controls{display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin-bottom:12px}
  .chips{display:flex; flex-wrap:wrap; gap:7px}
  .chip{font-size:12.5px; border:1px solid var(--line); border-radius:999px; padding:4px 10px; cursor:pointer; color:var(--ink2); user-select:none; background:var(--surface)}
  .chip input{position:absolute; opacity:0; width:0; height:0}
  .chip.on{background:var(--accent); border-color:var(--accent); color:#fff}
  .chip:focus-within{outline:2px solid var(--accent); outline-offset:2px}
  .plot{overflow-x:auto}
  .empty{color:var(--muted); font-size:14px; padding:22px 4px}
  #spec{width:100%; height:200px; display:block; border-radius:6px; background:#08061a}
  .specrow{display:flex; gap:8px; align-items:stretch; margin:2px 0 11px}
  .freqax{display:flex; flex-direction:column; justify-content:space-between; width:40px; text-align:right; font-size:10px; color:var(--muted); padding:2px 2px}
  .canvholder{position:relative; flex:1; min-width:0}
  .playhead{position:absolute; top:0; bottom:0; left:0; width:2px; background:rgba(255,255,255,.85); display:none; pointer-events:none}
  .audioctl{display:flex; align-items:center; gap:9px; flex-wrap:wrap}
  .pbtn{background:var(--accent); color:#fff; border:none; border-radius:7px; padding:6px 15px; font-size:13px; cursor:pointer}
  .pbtn:hover{filter:brightness(1.07)}
  .stepbtn{background:var(--surface); border:1px solid var(--line); color:var(--ink2); border-radius:6px; width:27px; height:27px; cursor:pointer; font-size:15px; line-height:1}
  .stepbtn:disabled{opacity:.4; cursor:default}
  .detidx{font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; min-width:30px; text-align:center}
  .audioinfo{font-size:13px; color:var(--ink2); margin-top:9px; min-height:18px}
  .audioinfo .mono{font-family:ui-monospace,"SFMono-Regular",Menlo,monospace; font-size:12px}
  .listen{cursor:pointer; text-decoration:underline dotted; text-underline-offset:2px}
  .listen:hover{color:var(--accent)}
  .audioset{display:flex; flex-wrap:wrap; align-items:center; gap:8px 10px; margin-top:12px; padding-top:11px; border-top:1px solid var(--grid); font-size:12.5px; color:var(--ink2)}
  .setbtn{background:var(--surface); border:1px solid var(--line); color:var(--ink); border-radius:7px; padding:5px 10px; font-size:12.5px; cursor:pointer}
  .setbtn:hover{border-color:var(--accent)}
  .setnote{color:var(--muted)} .setsep{color:var(--line)}
  .setlbl{display:flex; align-items:center; gap:6px}
  .urlinput{font:inherit; font-size:12px; padding:4px 7px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--ink); width:118px}
  table{border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums}
  thead th{text-align:right; color:var(--muted); font-weight:600; padding:7px 10px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--surface)}
  thead th:first-child, tbody td:first-child{text-align:left}
  tbody td{padding:6px 10px; border-bottom:1px solid var(--grid)}
  tbody tr:hover{background:color-mix(in srgb, var(--accent) 7%, transparent)}
  .tableScroll{max-height:440px; overflow:auto; border:1px solid var(--grid); border-radius:8px}
  footer{color:var(--muted); font-size:12.5px; margin-top:26px; line-height:1.7}
  footer a{color:var(--accent)}
  @media (max-width:720px){ .tiles{grid-template-columns:repeat(2,1fr)} h1{font-size:27px} .scopehint{display:none} }
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div>
      <h1 class="display">Dawn&nbsp;Chorus</h1>
      <div class="sub" id="subline"></div>
    </div>
    <div class="hbtns">
      <button class="theme" id="addBtn" aria-haspopup="dialog">＋ Add data</button>
      <button class="theme" id="setBtn" aria-haspopup="dialog">⚙ Settings</button>
      <button class="theme" id="guideBtn" aria-haspopup="dialog">Guide</button>
      <button class="theme" id="theme" aria-label="Toggle light or dark theme">◐ Theme</button>
    </div>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="topline">
  <div class="scopebar">
    <div class="scoperow">
      <span class="eyebrow">Time&nbsp;scope</span>
      <label class="ctl">Aggregate
        <select id="aggSel">
          <option value="day">Daily</option>
          <option value="week">Weekly</option>
          <option value="month">Monthly</option>
          <option value="year">Yearly</option>
        </select>
      </label>
      <label class="ctl grow">Period
        <input type="range" id="periodSlider" min="0" max="0" step="1" value="0" aria-label="Period">
      </label>
      <span class="periodlabel" id="periodLabel"></span>
      <span class="scopehint">scopes every chart below</span>
    </div>
    <div class="scoperow">
      <span class="eyebrow">Time&nbsp;axis</span>
      <span class="seg" id="xmodeSeg" role="group" aria-label="Time axis">
        <button type="button" data-xmode="dawn" aria-pressed="true">from dawn</button>
        <button type="button" data-xmode="clock" aria-pressed="false">clock</button>
      </span>
      <span class="scopehint">x-axis on the morning-profile charts</span>
    </div>
  </div>

  <section class="card audio" id="audioCard" hidden>
    <h2 class="display">Listen</h2>
    <p class="lead" id="audioLead"></p>
    <div id="specWrap" hidden>
      <div class="specrow">
        <div class="freqax" id="freqax"><span></span><span></span><span></span></div>
        <div class="canvholder"><canvas id="spec"></canvas><div class="playhead" id="playhead"></div></div>
      </div>
      <div class="audioctl">
        <button class="pbtn" id="playBtn">▶&nbsp;Play</button>
        <button class="stepbtn" id="prevDet" title="previous detection">‹</button>
        <span class="detidx" id="detIdx"></span>
        <button class="stepbtn" id="nextDet" title="next detection">›</button>
      </div>
    </div>
    <div class="audioinfo" id="audioInfo"></div>
    <div class="audioset">
      <input type="file" id="dirPick" webkitdirectory directory multiple hidden>
      <button class="setbtn" id="dirBtn">📁 Recordings folder…</button>
      <span class="setnote" id="dirStatus"></span>
      <span class="setsep">·</span>
      <label class="setlbl">served base <input type="text" id="urlBase" class="urlinput" spellcheck="false"></label>
      <span class="setsep">·</span>
      <label class="setlbl">colours
        <select id="specMode"><option value="color">Colour</option><option value="bw">B&amp;W</option></select>
      </label>
    </div>
  </section>
  </div>

  <div class="chapters">

  <section class="chapter" id="ch-morning">
    <div class="rail">
      <span class="chwhen">dawn &minus;2h</span>
      <h2 class="chtitle">The first voices</h2>
      <p class="chq">Who starts singing, and how long do they keep going?</p>
      <p class="chfind" id="find-morning"></p>
    </div>
    <div class="chbody">
      <section class="card" data-card="timeline">
        <h2 class="display">Who sings when</h2>
        <p class="lead">Each bar spans a species' vocal activity &mdash; onset (5th percentile of detection
          times) to offset (95th) &mdash; in minutes from <span class="tag">civil dawn</span> (dashed line),
          taken as the <em>median across the scoped period's mornings</em>. Darker bars are sung more
          continuously; the tick marks the median busiest minute.</p>
        <div class="plot" id="chart-timeline"></div>
      </section>
    </div>
  </section>

  <section class="chapter" id="ch-shape">
    <div class="rail">
      <span class="chwhen">dawn</span>
      <h2 class="chtitle">The shape of a morning</h2>
      <p class="chq">How does the chorus build and fade across the window?</p>
      <p class="chfind" id="find-shape"></p>
    </div>
    <div class="chbody pair">
      <section class="card" data-card="ecdf">
        <h2 class="display">Cumulative call distributions</h2>
        <p class="lead">The empirical CDF <em>F(t)</em> &mdash; share of a species' detections that have
          occurred by each minute &mdash; averaged across the period's mornings (each morning a replicate).
          Onset reads where a curve crosses 0.05, median song-time at 0.5, offset at 0.95.</p>
        <div class="controls"><div class="chips" id="ecdfChips"></div></div>
        <div class="plot" id="chart-ecdf"></div>
      </section>
      <section class="card" data-card="heat">
        <h2 class="display">Occupancy across the morning</h2>
        <p class="lead">Species &times; solar-minute. Colour is the fraction of the scoped period's mornings
          a species was detected in each 5-minute bin &mdash; slide the time scope to watch the chorus shift.</p>
        <div class="plot" id="chart-heat"></div>
      </section>
    </div>
  </section>

  <section class="chapter" id="ch-season">
    <div class="rail">
      <span class="chwhen">week to week</span>
      <h2 class="chtitle">Through the season</h2>
      <p class="chq">Does a species shift earlier or later as the year turns?</p>
      <p class="chfind" id="find-season"></p>
    </div>
    <div class="chbody">
      <section class="card" data-card="season">
        <h2 class="display">Onset through the season</h2>
        <p class="lead">Each dot is one species on one morning: when it started, against the date.
          This is the seasonal question &mdash; <em>does a species shift earlier or later as the year
          turns?</em> Trend lines appear for species with &ge;4 mornings. Unlike the charts above this
          one ignores the time scope and always shows <em>every</em> morning, because a season is
          what it is measuring.</p>
        <div class="controls"><div class="chips" id="seasonChips"></div></div>
        <div class="plot" id="chart-season"></div>
      </section>
    </div>
  </section>

  <section class="chapter" id="ch-weather">
    <div class="rail">
      <span class="chwhen">each morning</span>
      <h2 class="chtitle">Against the weather</h2>
      <p class="chq">Do they start earlier when it is warm, later when it rains?</p>
      <p class="chfind" id="find-weather"></p>
    </div>
    <div class="chbody pair">
      <section class="card" data-card="temp">
        <h2 class="display">Onset vs.&nbsp;temperature</h2>
        <p class="lead">Each dot is a species on one morning: its onset (minutes from <span class="tag">civil
          dawn</span>) against the temperature at dawn &mdash; the founding question, <em>does the chorus start
          earlier on warm mornings?</em> A downward trend says yes. Per-species trend lines appear once a
          species has &ge;4 mornings in the period; treat pooled patterns cautiously (weather confounds season).</p>
        <div class="plot" id="chart-temp"></div>
      </section>
      <section class="card" data-card="rain">
        <h2 class="display">Onset vs.&nbsp;rain</h2>
        <p class="lead">Onset against total rainfall over the morning window (mm). Points sit at 0 on dry
          mornings; a rightward shift on wet mornings would show birds starting later in the rain.</p>
        <div class="plot" id="chart-rain"></div>
      </section>
    </div>
  </section>

  <section class="chapter" id="ch-species">
    <div class="rail">
      <span class="chwhen">all mornings</span>
      <h2 class="chtitle">Every species</h2>
      <p class="chq">What did each one actually do?</p>
      <p class="chfind" id="find-species"></p>
    </div>
    <div class="chbody">
      <section class="card" data-card="table">
        <h2 class="display">Per-species table</h2>
        <p class="lead">Aggregated over the scoped period: mornings present, total detections, and median
          onset/offset/span/peak/occupancy (minutes from civil dawn).</p>
        <div class="tableScroll"><table id="tbl"></table></div>
      </section>
    </div>
  </section>
  </div>

  <footer id="foot"></footer>
</div>

<div class="modal" id="setModal" hidden>
  <div class="dialog" role="dialog" aria-modal="true" aria-label="Settings">
    <button class="close" id="setClose" aria-label="Close">&times;</button>
    <h2 class="display">Settings</h2>

    <div class="setrow">
      <b>Time axis</b>
      <label><input type="radio" name="xmode" value="dawn" checked>
        <span>Minutes from civil dawn
          <small>Comparable across the season: dawn drifts by more than an hour between
            midsummer and autumn, so a fixed clock hides the shift you are looking for.</small></span></label>
      <label><input type="radio" name="xmode" value="clock">
        <span>Time of day
          <small>Wall-clock, using the median civil dawn of the mornings in scope. Easier to
            match against a recording, but a period spanning weeks is only approximate.</small></span></label>
    </div>

    <div class="setrow">
      <b>Cards</b>
      <label><input type="checkbox" id="autoCollapse" checked>
        <span>Collapse thin charts automatically
          <small>Hides a chart when the scoped period gives it fewer than 8 points &mdash;
            too few to read a relationship from. Opening one yourself always wins.</small></span></label>
      <div class="row" style="gap:8px; margin-top:4px">
        <button class="theme" id="expandAll">Expand all</button>
        <button class="theme" id="collapseAll">Collapse all</button>
      </div>
    </div>

    <p class="fine">Settings are remembered in this browser only &mdash; they travel with neither
      the data nor the published page.</p>
  </div>
</div>

<div class="modal" id="guide" hidden>
  <div class="dialog" role="dialog" aria-modal="true" aria-label="User's guide">
    <button class="close" id="guideClose" aria-label="Close guide">×</button>
    <h2 class="display">Using this dashboard</h2>
    <p>Every chart plots a species' morning singing in <b>solar time</b> — minutes from
      <span class="tag">civil dawn</span> (the dashed line at 0) — so mornings weeks apart line up.</p>

    <h3>Time scope</h3>
    <p><b>Aggregate</b> (Daily / Weekly / Monthly / Yearly) plus the <b>Period</b> slider pick a set of
      mornings, and <b>every chart recomputes</b> for them. Slide through time to watch the chorus shift.</p>

    <h3>The charts</h3>
    <ul>
      <li><b>Who sings when</b> — each bar runs from a species' <b>onset</b> (when 5% of its detections
        have occurred) to its <b>offset</b> (95%), median across the period's mornings. Darker = sung
        more continuously; the tick marks the busiest minute.</li>
      <li><b>Cumulative call distributions</b> — the curve <em>F(t)</em> is the share of a species'
        detections by each minute; it crosses 0.05 at onset, 0.5 at median song-time, 0.95 at offset.
        Toggle species with the chips.</li>
      <li><b>Occupancy</b> — species &times; 5-minute bin; colour is the fraction of the period's mornings
        that species was detected in that bin.</li>
      <li><b>Per-species table</b> — the same measures, aggregated over the period.</li>
    </ul>

    <h3>Listen &amp; spectrograms <span class="muted">(local dashboard, Daily scope)</span></h3>
    <p class="muted">Click-to-listen works when you run the dashboard on the machine that holds the
      recordings — not on the published site, where the audio stays local.</p>
    <ul>
      <li><b>Click a chart</b> — it snaps to the nearest call and shows a ~5-second <b>spectrogram</b>
        with each detected species boxed and labelled (name + confidence).</li>
      <li><b>Click a species</b> in the table to jump to its best example.</li>
      <li><b>‹ ›</b> step through calls, <b>▶ Play</b> plays the clip, <b>Colour / B&amp;W</b> switches
        the spectrogram style.</li>
      <li><b>Recordings:</b> click <b>Recordings folder…</b> to point at your local WAVs (works offline,
        no server), or set the <b>served base</b> URL if you're serving them.</li>
    </ul>

    <h3>Good to know</h3>
    <ul>
      <li>BirdNET works in 3-second windows and doesn't separate song from call, so "span" is a
        vocal-activity span, not a song-bout length.</li>
      <li>Charts count detections at &ge;0.5 confidence; spectrogram labels go lower to show more calls.
        Many faint calls stay unlabelled — BirdNET didn't clear the bar.</li>
      <li>Detectability depends on distance, wind, and the mic — keep the recorder fixed across the
        season for comparisons to hold.</li>
    </ul>

    <h3>Adding your own data</h3>
    <p>New mornings, or a whole new site? Use the <span class="tag">＋ Add data</span> button in the
      header — it walks through the short local step. Recordings stay on your machine; only the
      detection times are published.</p>
  </div>
</div>

<div class="modal" id="addModal" hidden>
  <div class="dialog" role="dialog" aria-modal="true" aria-label="Add data">
    <button class="close" id="addClose" aria-label="Close">&times;</button>
    <h2 class="display">Add data</h2>
    <p>This site is <b>static</b> — the charts are precomputed files, so there's no upload in the
      browser. You add data with a short step on the machine that has the recordings; they never leave
      it (only detection times are published). About a minute, start to finish.</p>

    <h3>1 &middot; New mornings for a site you already have</h3>
    <p>Drop the new recordings in your audio folder, then detect on just the new files:</p>
    <pre>python tools/track.py process --audio data --lat 42.537278 --lon -72.531694 --tz America/New_York</pre>
    <p>Regenerate this site's data and publish it (commits &amp; pushes &rarr; live in ~1&nbsp;min):</p>
    <pre>python tools/publish.py --slug montague --name "North St, Montague, MA" --from-analyzer data/results --lat 42.537278 --lon -72.531694 --tz America/New_York --push</pre>

    <h3>2 &middot; A whole new site</h3>
    <p>Same commands with a new <span class="tag">--slug</span>, <span class="tag">--name</span>, and
      your station's coordinates — it's added to the picker automatically.</p>
    <p class="muted">Contributing to someone else's instance? Run <span class="tag">build_payloads.py</span>
      for your site and open a pull request with the new <span class="tag">site/data/&lt;slug&gt;.json</span>
      (and the updated <span class="tag">sites.json</span>) — or send that one file to the maintainer.
      Full details are in the project README.</p>
  </div>
</div>

<!--__VIEWER_BOOTSTRAP__-->
<script type="application/json" id="data">/*__DATA__*/</script>
<script>
function boot(DATA){
const {meta, summary, counts, day_keys} = DATA;
const root = document.documentElement;
const css = v => getComputedStyle(root).getPropertyValue(v).trim();
const fmt = (x, d=0) => (x==null || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const seriesColors = () => ["--s1","--s2","--s3","--s4","--s5","--s6","--s7","--s8"].map(css);

const GRID = meta.grid, NB = GRID.length, HALF = meta.bin/2;
const DAYIDX = {}; meta.days.forEach((d,i)=> DAYIDX[d]=i);
const SPIDX = {}; meta.species.forEach((s,i)=> SPIDX[s]=i);

// per (dayIdx|spIdx) -> dense bin-count array. The single building block for every chart.
const BDS = {};
counts.forEach(([di,si,bi,c])=>{ const k=di+"|"+si; (BDS[k] || (BDS[k]=new Float64Array(NB)))[bi]=c; });

const AUDIO = DATA.audio || null, DAWN = DATA.dawn || {}, CLIPS = DATA.clips || {};
const LABELSP = meta.label_species || [], DETS = DATA.dets || {};
const CLIP_SEC = 5.5, PRE = 0.5, SNAP_MIN = 2;            // ~5s clip; snap a click to a detection within 2 min
let ABASE = localStorage.getItem("dc_audio_base") || meta.audio_base || "../data";
let specMode = localStorage.getItem("dc_spec_mode") || "color";
const audioFiles = {};                                    // basename -> File (from a picked local folder)
const hasAudio = () => AUDIO && Object.keys(AUDIO).length > 0;
const headerCache = {};
let actx=null, srcNode=null, playRAF=null, cur=null;

function secToClock(sec){ sec=Math.max(0,Math.round(sec)); const h=Math.floor(sec/3600), m=Math.floor(sec%3600/60), s=sec%60;
  return [h,m,s].map(v=>String(v).padStart(2,"0")).join(":"); }
function setAudioInfo(html){ const e=document.getElementById("audioInfo"); if(e) e.innerHTML=html; }
function showSpec(on){ const w=document.getElementById("specWrap"); if(w) w.hidden=!on; }

function audioUrl(name){ return ABASE.replace(/\/+$/,"") + "/" + encodeURIComponent(name); }
async function getBytes(name, s, e){                       // e inclusive; local picked File, else served fetch
  const file=audioFiles[name];
  if(file) return new DataView(await file.slice(s, e+1).arrayBuffer());
  const r=await fetch(audioUrl(name), {headers:{Range:`bytes=${s}-${e}`}});
  if(!(r.ok || r.status===206)) throw new Error("HTTP "+r.status);
  return new DataView(await r.arrayBuffer()); }
async function wavHeader(name){
  if(headerCache[name]) return headerCache[name];
  const dv=await getBytes(name,0,65535); let off=12, fmt=null, dataOff=null, dataLen=null;
  while(off+8<=dv.byteLength){
    const id=String.fromCharCode(dv.getUint8(off),dv.getUint8(off+1),dv.getUint8(off+2),dv.getUint8(off+3));
    const sz=dv.getUint32(off+4,true);
    if(id==="fmt "){ fmt={channels:dv.getUint16(off+10,true), sampleRate:dv.getUint32(off+12,true), bits:dv.getUint16(off+22,true)}; }
    else if(id==="data"){ dataOff=off+8; dataLen=sz; break; }
    off+=8+sz+(sz&1); }
  const h={channels:fmt&&fmt.channels, sampleRate:fmt&&fmt.sampleRate, bits:fmt&&fmt.bits, dataOff, dataLen};
  headerCache[name]=h; return h; }
async function loadPcm(name, startSec, durSec){
  const h=await wavHeader(name);
  if(h.dataOff==null || h.bits!==16) throw new Error("need 16-bit WAV");
  const bpf=h.channels*2, sr=h.sampleRate;
  let s=h.dataOff+Math.floor(startSec*sr)*bpf, e=h.dataOff+Math.ceil((startSec+durSec)*sr)*bpf-1;
  e=Math.min(e, h.dataOff+h.dataLen-1); if(s<h.dataOff) s=h.dataOff;
  const dv=await getBytes(name,s,e); const n=Math.floor(dv.byteLength/bpf); const out=new Float32Array(n);
  for(let i=0;i<n;i++) out[i]=dv.getInt16(i*bpf,true)/32768;         // channel 0
  return {samples:out, sampleRate:sr}; }

function fft(re, im){ const n=re.length;
  for(let i=1,j=0;i<n;i++){ let bit=n>>1; for(;j&bit;bit>>=1) j^=bit; j^=bit;
    if(i<j){ const tr=re[i]; re[i]=re[j]; re[j]=tr; const ti=im[i]; im[i]=im[j]; im[j]=ti; } }
  for(let len=2;len<=n;len<<=1){ const ang=-2*Math.PI/len, wr=Math.cos(ang), wi=Math.sin(ang);
    for(let i=0;i<n;i+=len){ let cr=1, ci=0;
      for(let k=0;k<len/2;k++){ const ar=re[i+k], ai=im[i+k],
          br=re[i+k+len/2]*cr-im[i+k+len/2]*ci, bi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
        re[i+k]=ar+br; im[i+k]=ai+bi; re[i+k+len/2]=ar-br; im[i+k+len/2]=ai-bi;
        const ncr=cr*wr-ci*wi; ci=cr*wi+ci*wr; cr=ncr; } } } }
const MAGMA=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function magma(v){ v=v>0?(v<1?v:1):0; const x=v*(MAGMA.length-1), i=Math.floor(x), f=x-i, a=MAGMA[i], b=MAGMA[Math.min(i+1,MAGMA.length-1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]; }
function gray(v){ v=v>0?(v<1?v:1):0; const g=Math.round(255*(1-Math.pow(v,0.85))); return [g,g,g]; }   // Merlin-style: dark ink on white
function drawSpec(samples, sr){
  const canvas=document.getElementById("spec"), holder=canvas.parentElement;
  const W=canvas.width=Math.max(320, holder.clientWidth), H=canvas.height=200;
  const N=1024, hop=256, win=new Float32Array(N);
  for(let i=0;i<N;i++) win[i]=0.5-0.5*Math.cos(2*Math.PI*i/(N-1));
  const frames=Math.max(1, Math.floor((samples.length-N)/hop)+1), bins=N/2,
        nyq=sr/2, maxF=Math.min(12000,nyq), maxBin=Math.max(1, Math.round(maxF/nyq*bins));
  const mag=new Float32Array(frames*maxBin), re=new Float32Array(N), im=new Float32Array(N);
  let mn=Infinity, mx=-Infinity;
  for(let fr=0;fr<frames;fr++){ const s0=fr*hop;
    for(let i=0;i<N;i++){ re[i]=(samples[s0+i]||0)*win[i]; im[i]=0; }
    fft(re,im);
    for(let b=0;b<maxBin;b++){ const m=Math.log10(1e-7+re[b]*re[b]+im[b]*im[b]); mag[fr*maxBin+b]=m; if(m<mn)mn=m; if(m>mx)mx=m; } }
  const rng=(mx-mn)||1, ctx=canvas.getContext("2d"), img=ctx.createImageData(W,H), paint=specMode==="bw"?gray:magma;
  for(let x=0;x<W;x++){ const fr=Math.min(frames-1, Math.floor(x/W*frames));
    for(let y=0;y<H;y++){ const b=Math.min(maxBin-1, Math.floor((1-y/H)*maxBin));
      let v=(mag[fr*maxBin+b]-mn)/rng; v=v>0?(v<1?v:1):0; v=Math.pow(v,1.4);
      const c=paint(v), idx=(y*W+x)*4; img.data[idx]=c[0]; img.data[idx+1]=c[1]; img.data[idx+2]=c[2]; img.data[idx+3]=255; } }
  ctx.putImageData(img,0,0);
  const ax=document.getElementById("freqax");
  if(ax){ ax.children[0].textContent=(maxF/1000).toFixed(0)+" kHz"; ax.children[1].textContent=(maxF/2000).toFixed(0); ax.children[2].textContent="0"; } }

function renderClip(){ if(cur && cur.clip){ drawSpec(cur.clip.samples, cur.clip.sampleRate); drawLabels(); } }
function drawLabels(){                                     // BirdNET 3s detections boxed + named over the clip
  if(!cur || !cur.file || !DETS[cur.day]) return;
  const canvas=document.getElementById("spec"), ctx=canvas.getContext("2d"), W=canvas.width, H=canvas.height;
  const f=cur.file, startOff=cur.startOff, dawnSec=DAWN[cur.day], SEG=3, dark=specMode!=="bw";
  const best={};                                          // one box per species (highest conf) in the window
  DETS[cur.day].forEach(d=>{ const ft=(dawnSec + d[0]*60) - f.s;
    if(ft+SEG<startOff || ft>startOff+CLIP_SEC) return;
    if(!best[d[1]] || d[2]>best[d[1]].c) best[d[1]]={ft, si:d[1], c:d[2]}; });
  const rows=Object.values(best).sort((a,b)=>a.ft-b.ft);
  ctx.font="600 11px system-ui,sans-serif"; ctx.textBaseline="top";
  rows.forEach((r,i)=>{ const x1=Math.max(1,(r.ft-startOff)/CLIP_SEC*W), x2=Math.min(W-1,(r.ft+SEG-startOff)/CLIP_SEC*W);
    ctx.strokeStyle=dark?"rgba(255,255,255,.85)":"rgba(0,0,0,.7)"; ctx.lineWidth=1.5; ctx.strokeRect(x1,2,x2-x1,H-4);
    const lab=`${LABELSP[r.si]} ${r.c.toFixed(2)}`, tw=ctx.measureText(lab).width, ly=3+(i%3)*15;
    ctx.fillStyle=dark?"rgba(0,0,0,.6)":"rgba(255,255,255,.82)"; ctx.fillRect(x1, ly, Math.min(tw+8, W-x1), 14);
    ctx.fillStyle=dark?"#fff":"#111"; ctx.fillText(lab, x1+4, ly+1); });
}

function playClip(){ if(!cur||!cur.clip) return;
  const ctx=actx||(actx=new (window.AudioContext||window.webkitAudioContext)());
  if(srcNode){ try{srcNode.stop()}catch(e){} }
  const {samples,sampleRate}=cur.clip, buf=ctx.createBuffer(1,samples.length,sampleRate); buf.copyToChannel(samples,0);
  srcNode=ctx.createBufferSource(); srcNode.buffer=buf; srcNode.connect(ctx.destination);
  const dur=samples.length/sampleRate, ph=document.getElementById("playhead"), t0=ctx.currentTime;
  ph.style.display="block"; srcNode.start(); cancelAnimationFrame(playRAF);
  (function tick(){ const el=(ctx.currentTime-t0)/dur; if(el>=1){ ph.style.display="none"; return; } ph.style.left=(el*100)+"%"; playRAF=requestAnimationFrame(tick); })();
  srcNode.onended=()=>{ ph.style.display="none"; }; }

const _distinct={};                                          // deduped, chronological calls per day (collapse overlap dupes)
function distinctDets(day){
  if(_distinct[day]) return _distinct[day];
  const arr=(DETS[day]||[]).slice().sort((a,b)=> (a[1]-b[1]) || (a[0]-b[0]));   // by species, then time
  const out=[]; let ls=-1, lt=-1e9;
  for(const d of arr){
    if(d[1]===ls && (d[0]-lt)*60 < 2){ if(d[2]>out[out.length-1][2]) out[out.length-1]=d.slice(); lt=d[0]; continue; }
    out.push(d.slice()); ls=d[1]; lt=d[0];
  }
  out.sort((a,b)=>a[0]-b[0]);
  return (_distinct[day]=out);
}

async function loadAt(day, list, pos){                        // load + draw list[pos]; enable/disable the ‹ › steps
  if(!list.length){ showSpec(false); setAudioInfo("No detections here."); return; }
  pos=Math.max(0, Math.min(pos, list.length-1));
  const e=list[pos], t=e[0], name=LABELSP[e[1]], c=e[2], files=AUDIO[day], dawnSec=DAWN[day];
  if(!files || dawnSec==null){ setAudioInfo(`No recording for ${day}.`); showSpec(false); return; }
  const abs=dawnSec + t*60, f=files.find(x=>x.d!=null && abs>=x.s && abs<x.s+x.d) || files.find(x=>abs>=x.s) || files[0];
  const startOff=Math.max(0, (abs - f.s) - CLIP_SEC/2);
  cur={day, list, pos, file:f, startOff};
  setAudioInfo(`loading… <span class="mono">${f.name}</span>`);
  try{ cur.clip=await loadPcm(f.name, startOff, CLIP_SEC);
    showSpec(true); renderClip();
    document.getElementById("detIdx").textContent=`${pos+1}/${list.length}`;
    document.getElementById("prevDet").disabled=pos<=0;
    document.getElementById("nextDet").disabled=pos>=list.length-1;
    setAudioInfo(`<strong>${name}</strong> ${c.toFixed(2)} · ${secToClock(abs)} local · ${Math.round(t)} min from dawn · <span class="mono">${f.name}</span>`);
  }catch(err){ showSpec(false); setAudioInfo(audioErr(err)); } }

function audioErr(e){ return `Couldn't load audio (${e.message}). Pick your recordings folder below, or run <code>python tools/serve.py</code> and open <code>/site/</code>.`; }

async function openAudioFor(name){                            // table species -> that species' calls, best first
  const S=scopedDays(); if(!hasAudio() || S.length!==1) return;
  const day=S[0], si=LABELSP.indexOf(name);
  document.getElementById("audioCard").scrollIntoView({behavior:"smooth", block:"nearest"});
  const list=distinctDets(day).filter(d=>d[1]===si);
  if(!list.length){ showSpec(false); setAudioInfo(`No detections for ${name} on ${day}.`); return; }
  let pos=0, bc=-1; list.forEach((d,i)=>{ if(d[2]>bc){ bc=d[2]; pos=i; } });   // start on the best example
  await loadAt(day, list, pos); }

async function openAudioAt(day, xMin){                        // click a chart -> nearest call; ‹ › scrub all calls
  document.getElementById("audioCard").scrollIntoView({behavior:"smooth", block:"nearest"});
  if(!AUDIO[day] || DAWN[day]==null) return;
  const list=distinctDets(day);
  if(!list.length){ showSpec(false); setAudioInfo(`No detections on ${day}.`); return; }
  let pos=0, best=Infinity; list.forEach((d,i)=>{ const diff=Math.abs(d[0]-xMin); if(diff<best){ best=diff; pos=i; } });
  await loadAt(day, list, pos); }

function reloadCur(){ if(cur && cur.list) loadAt(cur.day, cur.list, cur.pos); }

function wireTimeClicks(el, S){
  if(!hasAudio() || S.length!==1) return;
  const p=el.firstElementChild; if(!p || !p.scale) return;
  const xs=p.scale("x"); if(!xs || !xs.invert) return;
  // A plot with a continuous colour legend is a <figure> holding the legend ramp <svg>
  // AND the plot <svg>; pick the largest so we don't wire the tiny legend by mistake.
  const svgs=p.matches("svg") ? [p] : [...p.querySelectorAll("svg")];
  if(!svgs.length) return;
  const area=s=>{ const r=s.getBoundingClientRect(); return r.width*r.height; };
  const svg=svgs.reduce((a,b)=> area(b)>area(a) ? b : a);
  svg.style.cursor="crosshair";
  svg.addEventListener("click", ev=>{
    const rect=svg.getBoundingClientRect();
    const iw=(svg.width && svg.width.baseVal && svg.width.baseVal.value) || rect.width;
    openAudioAt(S[0], xs.invert((ev.clientX-rect.left)/rect.width*iw));
  });
}
function updateAudioCard(S){
  const card=document.getElementById("audioCard"); if(!card) return;
  if(!hasAudio()){ card.hidden=true; return; }
  card.hidden=false;
  document.getElementById("audioLead").innerHTML = S.length===1
    ? `Click a chart (snaps to the nearest call) or a table species to see its <strong>spectrogram</strong> on <strong>${S[0]}</strong>; use <strong>‹ ›</strong> to step through calls, ▶ to play.`
    : `Set the time scope to a single day (<strong>Daily</strong>) to view spectrograms.`;
}

const median = a => { const s=a.filter(v=>v!=null).sort((x,y)=>x-y), n=s.length;
  return n ? (n%2 ? s[(n-1)/2] : (s[n/2-1]+s[n/2])/2) : null; };

const colorFor = {};
function refreshColors(){ const c=seriesColors();
  meta.top_species.forEach((s,i)=> colorFor[s]=c[i%8]); }
function plotStyle(){ return {background:"transparent", color:css("--ink"), fontSize:"12.5px"}; }
function W(el){ return Math.max(300, el.clientWidth || 900); }

/* ---- global time scope ---- */
const aggSel = document.getElementById("aggSel");
const slider = document.getElementById("periodSlider");
let periods = [];
function periodsFor(level){ const out=[], seen=new Set();
  meta.days.forEach(d=>{ const k=day_keys[d][level]; if(!seen.has(k)){ seen.add(k); out.push(k); } });
  return out; }
function scopedDays(){ const level=aggSel.value, pkey=periods[Math.min(+slider.value, periods.length-1)] ?? periods[0];
  return meta.days.filter(d=>day_keys[d][level]===pkey); }
function rebuildPeriods(){ periods = periodsFor(aggSel.value);
  slider.min=0; slider.max=Math.max(0, periods.length-1); slider.value=periods.length-1;
  slider.disabled = periods.length<=1; renderAll(); }

function renderSubline(){ const m=meta, loc=(m.lat!=null&&m.lon!=null)?`${m.lat.toFixed(2)}, ${m.lon.toFixed(2)}`:"";
  document.getElementById("subline").innerHTML =
    `${m.days.length} morning${m.days.length>1?"s":""} &middot; ${loc} &middot; ${m.tz||""}`; }

function renderTiles(){ const m=meta, e=m.earliest;
  const tiles=[["Mornings", `${m.days[0]||"—"}${m.days.length>1?" → "+m.days[m.days.length-1]:""}`],
    ["Species", m.n_species],
    ["Detections", m.n_detections.toLocaleString()+` <small>&ge;${m.min_confidence} conf</small>`],
    ["Earliest onset", e ? `${fmt(e.onset,0)}<small> min &middot; ${e.name}</small>` : "—"]];
  document.getElementById("tiles").innerHTML = tiles.map(([k,v])=>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join(""); }

function updateScopeLabel(S){ const pkey=periods[Math.min(+slider.value, periods.length-1)] ?? periods[0];
  document.getElementById("periodLabel").textContent = `${pkey||"—"} · ${S.length} morning${S.length!==1?"s":""}`; }

/* ---- charts (all take the scoped set of mornings S) ---- */
// ---- settings -------------------------------------------------------------------------
// Persisted per browser. Kept in one object so adding an option later is a one-line change
// here plus a control in the Settings dialog, rather than another ad-hoc global.
const SET_KEY="dc.settings";
const SETTINGS=Object.assign({xmode:"dawn", autoCollapse:true},
  (()=>{ try{ return JSON.parse(localStorage.getItem(SET_KEY)||"{}"); }catch(e){ return {}; } })());
function saveSettings(){ try{ localStorage.setItem(SET_KEY, JSON.stringify(SETTINGS)); }catch(e){} }

// Cards the user has explicitly opened or closed. Auto-collapse must never override a
// deliberate choice, so it only applies to cards absent from here.
const CARD_KEY="dc.cards";
const CARDS=(()=>{ try{ return JSON.parse(localStorage.getItem(CARD_KEY)||"{}"); }catch(e){ return {}; } })();
function saveCards(){ try{ localStorage.setItem(CARD_KEY, JSON.stringify(CARDS)); }catch(e){} }

function cardEl(key){ return document.querySelector(`.card[data-card="${key}"]`); }
function setCollapsed(key, on, explicit){
  const el=cardEl(key); if(!el) return;
  el.classList.toggle("collapsed", !!on);
  if(explicit){ CARDS[key]=!!on; saveCards(); }
}
// Auto-collapse a chart the scope has left too thin to read. `n` is the number of points
// the chart would draw, so this follows the aggregation level for free.
const THIN=8;
function autoThin(key, n){
  if(key in CARDS) return;                                   // the user has decided; leave it
  const el=cardEl(key); if(!el) return;
  const thin = SETTINGS.autoCollapse && n < THIN;
  el.classList.toggle("collapsed", thin);
  const h=el.querySelector("h2"); let why=h && h.querySelector(".why");
  if(h){ if(thin){ if(!why){ why=document.createElement("span"); why.className="why"; h.appendChild(why); }
                   why.textContent = n ? `only ${n} point${n===1?"":"s"} in this period` : "no data in this period"; }
         else if(why) why.remove(); }
}

// ---- time axis ------------------------------------------------------------------------
// Everything is stored as minutes from civil dawn. Clock mode shifts by the scope's MEDIAN
// dawn: dawn moves through the season, so one period can only have one honest offset.
// (DAWN itself is declared with the audio globals above -- the clip player needs it too.)
let XOFF = 0;                                                // minutes to add for clock mode
function setXOffset(S){
  if(SETTINGS.xmode!=="clock"){ XOFF=0; return; }
  const secs=S.map(d=>DAWN[d]).filter(v=>v!=null).sort((a,b)=>a-b);
  XOFF = secs.length ? secs[Math.floor(secs.length/2)]/60 : 0;
}
const xv = m => m==null ? null : m + XOFF;                   // datum -> displayed x
const clockMode = () => SETTINGS.xmode==="clock" && XOFF>0;
function hhmm(min){ let t=((Math.round(min)%1440)+1440)%1440;
  return String(Math.floor(t/60)).padStart(2,"0")+":"+String(t%60).padStart(2,"0"); }
function xAxis(extra){
  const base = clockMode()
    ? {label:"time of day →", domain:meta.window.map(xv), tickFormat:hhmm}
    : {label:"minutes from civil dawn →", domain:meta.window};
  return Object.assign(base, extra||{});
}
const xTitle = v => clockMode() ? hhmm(v) : fmt(v,0)+" min";
function yOnsetLabel(){ return clockMode() ? "onset — time of day ↑" : "onset — min from civil dawn ↑"; }

function ML(el, wide){ const w=W(el);
  return w<520 ? (wide?96:40) : (wide?172:52); }
// Height from width, clamped. A value chart with a FIXED height gets letterboxed as the
// window widens, which visually flattens every slope -- and these charts exist to judge
// whether a slope is real. Scaling keeps the aspect honest at any width; the clamp stops
// a phone getting a sliver and an ultrawide getting a billboard.
function ratioH(el, ratio, lo, hi){
  return Math.round(Math.min(hi, Math.max(lo, W(el)*ratio))); }
function spName(n, el){ const w=W(el); const cap = w<520 ? 13 : 99;
  return n.length>cap ? n.slice(0,cap-1)+'…' : n; }

function aggBySpecies(S){ const set=new Set(S), g={};
  summary.forEach(r=>{ if(set.has(r.date)){ (g[r.name] || (g[r.name]=[])).push(r); } });
  return g; }

function renderTimeline(S){
  const el=document.getElementById("chart-timeline"); el.innerHTML="";
  const g=aggBySpecies(S);
  const rows=Object.keys(g).map(name=>{ const rs=g[name].filter(r=>r.onset!=null);
    if(!rs.length) return null;
    return {name, onset:median(rs.map(r=>r.onset)), offset:median(rs.map(r=>r.offset)),
      peak:median(rs.map(r=>r.peak)), occ:median(rs.map(r=>r.occ)),
      n:rs.reduce((s,r)=>s+r.n,0), mornings:rs.length};
  }).filter(Boolean).sort((a,b)=>a.onset-b.onset);
  if(!rows.length){ el.innerHTML='<p class="empty">No species cleared the onset threshold in this period.</p>'; return; }
  const names=rows.map(d=>d.name);
  el.append(Plot.plot({ width:W(el), height:names.length*26+80, marginLeft:ML(el,true), marginRight:26, style:plotStyle(),
    x:xAxis({grid:true}), y:{domain:names, label:null, tickFormat:n=>spName(n,el)},
    color:{type:"linear", domain:[0,1], range:[css("--seq-lo"),css("--seq-hi")], legend:true, label:"median occupancy"},
    marks:[
      Plot.ruleX([xv(0)], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
      Plot.barX(rows, {x1:d=>xv(d.onset), x2:d=>xv(d.offset), y:"name", fill:"occ", rx:3, insetTop:5, insetBottom:5}),
      Plot.tickX(rows, {x:d=>xv(d.peak), y:"name", stroke:css("--ink"), strokeWidth:2, strokeOpacity:.85}),
      Plot.dot(rows, {x:d=>xv(d.onset), y:"name", r:3.4, fill:css("--ink"), stroke:css("--surface"), strokeWidth:1}),
      Plot.text(rows, {x:d=>xv(d.offset), y:"name", text:d=>d.n, dx:8, textAnchor:"start", fill:css("--muted"), fontSize:10}),
      Plot.tip(rows, Plot.pointerY({x:d=>xv(d.onset), y:"name",
        title:d=>`${d.name}\nonset ${xTitle(d.onset)}  offset ${xTitle(d.offset)}\nspan ${fmt(d.offset-d.onset,0)} min · occ ${fmt(d.occ,2)}\n${d.n} detections over ${d.mornings} morning(s)`}))
    ]}));
  wireTimeClicks(el, S);
}

function renderEcdf(S){
  const el=document.getElementById("chart-ecdf"); el.innerHTML="";
  const sel=[...document.querySelectorAll(".ecdf-sp:checked")].map(c=>c.value);
  const dIdx=S.map(d=>DAYIDX[d]);
  const lines=[];
  sel.forEach(name=>{ const si=SPIDX[name]; if(si==null) return;
    const acc=new Float64Array(NB); let ncur=0;
    for(const di of dIdx){ const a=BDS[di+"|"+si]; if(!a) continue;
      let tot=0; for(let b=0;b<NB;b++) tot+=a[b];
      if(tot<5) continue;                                   // library floor: >=5 detections/morning
      let cum=0; for(let b=0;b<NB;b++){ cum+=a[b]; acc[b]+=cum/tot; } ncur++; }
    if(ncur) for(let b=0;b<NB;b++) lines.push({name, t:GRID[b], F:acc[b]/ncur}); });
  if(!lines.length){ el.innerHTML='<p class="empty">No selected species had enough detections in this period.</p>'; return; }
  const shown=[...new Set(lines.map(d=>d.name))];
  el.append(Plot.plot({ width:W(el), height:ratioH(el,.55,360,520), marginLeft:ML(el,false), marginRight:20, style:plotStyle(),
    x:xAxis({grid:true}),
    y:{label:"↑ cumulative share of detections", domain:[0,1], ticks:[0,.25,.5,.75,1], grid:true},
    color:{domain:shown, range:shown.map(s=>colorFor[s]), legend:true},
    marks:[
      Plot.ruleY([0.05,0.5,0.95], {stroke:css("--grid")}),
      Plot.ruleX([xv(0)], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
      Plot.line(lines, {x:d=>xv(d.t), y:"F", stroke:"name", strokeWidth:2, tip:true}),
    ]}));
  wireTimeClicks(el, S);
}

function renderHeat(S){
  const el=document.getElementById("chart-heat"); el.innerHTML="";
  const names=meta.species, dIdx=S.map(d=>DAYIDX[d]);
  if(!names.length || !dIdx.length){ el.innerHTML='<p class="empty">Not enough data for the heatmap.</p>'; return; }
  const rows=[];
  for(let si=0; si<names.length; si++) for(let bi=0; bi<NB; bi++){
    let pres=0; for(const di of dIdx){ const a=BDS[di+"|"+si]; if(a && a[bi]>0) pres++; }
    if(pres>0) rows.push({name:names[si], t:GRID[bi], occ:pres/dIdx.length}); }
  el.append(Plot.plot({ width:W(el), height:names.length*20+80, marginLeft:ML(el,true), marginRight:26, style:plotStyle(),
    x:xAxis({grid:false}), y:{domain:names, label:null, tickFormat:n=>spName(n,el)},
    color:{type:"linear", domain:[0,1], range:[css("--seq-lo"),css("--seq-hi")], legend:true, label:"fraction of the period's mornings singing"},
    marks:[
      Plot.rect(rows, {x1:d=>xv(d.t-HALF), x2:d=>xv(d.t+HALF), y:"name", fill:"occ", inset:0.5,
        title:d=>`${d.name}\n${xTitle(d.t)} · ${Math.round(d.occ*100)}% of mornings`}),
      Plot.ruleX([xv(0)], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"})
    ]}));
  wireTimeClicks(el, S);
}

function renderWeather(S, key, elId, xlabel, card){
  const el=document.getElementById(elId); el.innerHTML="";
  const wx = DATA.weather || {};
  if(!S.some(d=> wx[d] && wx[d][key]!=null)){
    autoThin(card, 0);
    el.innerHTML='<p class="empty">No weather for this period yet &mdash; it fills in when mornings are generated with a network connection.</p>'; return; }
  const set=new Set(S), pts=[];
  summary.forEach(r=>{ if(!set.has(r.date) || r.onset==null) return;
    const w=wx[r.date]; if(!w || w[key]==null) return;
    pts.push({name:r.name, x:w[key], y:r.onset}); });
  autoThin(card, pts.length);
  if(!pts.length){ el.innerHTML='<p class="empty">No onsets to compare in this period.</p>'; return; }
  // per-species regression only where a species spans >=4 distinct mornings (else the line is meaningless)
  const xs={}; pts.forEach(p=>{ (xs[p.name]=xs[p.name]||new Set()).add(p.x); });
  const fitPts=pts.filter(p=> xs[p.name].size>=4 && colorFor[p.name]);
  const marks=[
    Plot.ruleY([xv(0)], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
    Plot.dot(pts, {x:"x", y:d=>xv(d.y), fill:d=>colorFor[d.name]||css("--muted"), r:3.6, fillOpacity:.85,
      stroke:css("--surface"), strokeWidth:.6, tip:true,
      title:d=>`${d.name}\n${xlabel}: ${fmt(d.x,1)}\nonset ${xTitle(d.y)}`}),
  ];
  if(fitPts.length && Plot.linearRegressionY)
    marks.push(Plot.linearRegressionY(fitPts, {x:"x", y:d=>xv(d.y), stroke:d=>colorFor[d.name], z:"name", strokeWidth:1.5, ci:0}));
  el.append(Plot.plot({ width:W(el), height:ratioH(el,.46,320,480), marginLeft:ML(el,false)+2, marginRight:20, style:plotStyle(),
    x:{label:xlabel, grid:true},
    y:Object.assign({label:yOnsetLabel(), grid:true}, clockMode()?{tickFormat:hhmm}:{}),
    marks}));
}

// Onset against DATE. Deliberately ignores the time scope: a season is the thing being
// measured, so restricting it to the scoped period would usually leave one or two points.
let SEASON_HL = null;                       // species under the pointer, or null

function renderSeason(){
  const el=document.getElementById("chart-season"); el.innerHTML="";
  const sel=new Set([...document.querySelectorAll(".season-sp:checked")].map(c=>c.value));
  const pts=[];
  summary.forEach(r=>{ if(r.onset==null || !sel.has(r.name)) return;
    pts.push({name:r.name, d:new Date(r.date+"T00:00:00"), date:r.date, y:r.onset, n:r.n}); });
  autoThin("season", pts.length);
  if(!pts.length){ SEASON_HL=null;
    el.innerHTML='<p class="empty">No species selected, or none has a defined onset yet.</p>'; return; }
  const shown=[...new Set(pts.map(p=>p.name))];
  if(SEASON_HL && !shown.includes(SEASON_HL)) SEASON_HL=null;   // deselected while hovered
  const days={}; pts.forEach(p=>{ (days[p.name]=days[p.name]||new Set()).add(p.date); });

  // Dim everything except the hovered species. Opacity rather than colour, so the legend
  // still reads and a faint dot keeps its position as context.
  const lit = d => !SEASON_HL || d.name===SEASON_HL;

  const marks=[
    Plot.ruleY([xv(0)], {stroke:css("--dawn"), strokeWidth:1.5, strokeDasharray:"4,3"}),
    Plot.dot(pts, {x:"d", y:d=>xv(d.y), fill:"name",
      r:d=>(SEASON_HL && d.name===SEASON_HL) ? 5.2 : 3.6,
      fillOpacity:d=>lit(d) ? .85 : .07,
      stroke:css("--surface"), strokeWidth:.6, tip:true,
      title:d=>`${d.name}\n${d.date}\nonset ${xTitle(d.y)}\n${d.n} detections`}),
  ];
  // One regression mark PER species. A single z-grouped mark would need strokeWidth as a
  // channel, which Plot does not accept on a line -- and per-mark constants are what make
  // the hovered line thicken at all.
  if(Plot.linearRegressionY) shown.forEach(sp=>{
    if(!days[sp] || days[sp].size<4) return;
    const on = !SEASON_HL || SEASON_HL===sp;
    marks.push(Plot.linearRegressionY(pts.filter(p=>p.name===sp), {
      x:"d", y:d=>xv(d.y), stroke:colorFor[sp]||css("--muted"),
      strokeWidth: SEASON_HL===sp ? 3.2 : 1.5,
      strokeOpacity: on ? .95 : .08, ci:0}));
  });

  const fig=Plot.plot({ width:W(el), height:ratioH(el,.34,380,620), marginLeft:ML(el,false)+4, marginRight:20, style:plotStyle(),
    x:{label:"morning →", grid:true, type:"time"},
    y:Object.assign({label:yOnsetLabel(), grid:true}, clockMode()?{tickFormat:hhmm}:{}),
    color:{domain:shown, range:shown.map(s=>colorFor[s]||css("--muted")), legend:true},
    marks});
  el.append(fig);
  wireSeasonHover(el, fig, pts);
}

// Highlight-on-hover. Nearest point in PIXEL space via the figure's own scales, rather
// than reading back the rendered SVG: Plot's element order is an implementation detail,
// and one extra mark would silently break any mapping built on it.
function wireSeasonHover(el, fig, pts){
  // With a legend, Plot returns a <figure> whose FIRST descendant <svg> is a legend
  // swatch, not the chart -- querySelector("svg") lands on a 15px square and no hover
  // ever fires. Take the figure's own direct child instead.
  const isSvg = n => n && n.tagName && n.tagName.toLowerCase()==="svg";
  const svg = isSvg(fig) ? fig : fig.querySelector(":scope > svg");
  if(!svg) return;
  const host = (fig && fig.scale) ? fig : (svg.scale ? svg : null);
  const sx = host && host.scale("x"), sy = host && host.scale("y");
  if(!sx || !sy || !sx.apply || !sy.apply) return;
  const proj=pts.map(p=>({name:p.name, cx:sx.apply(p.d), cy:sy.apply(xv(p.y))}))
                .filter(q=>isFinite(q.cx) && isFinite(q.cy));
  if(!proj.length) return;
  const R2=44*44;                                   // generous: species, not individual dots
  function pick(ev){
    const r=svg.getBoundingClientRect();
    // The SVG can be laid out narrower than its coordinate width (max-width:100%), so
    // scale the pointer into chart coordinates rather than assuming 1:1.
    const kx = r.width ? (svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width
                          ? svg.viewBox.baseVal.width / r.width : 1) : 1;
    const ky = r.height ? (svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height
                           ? svg.viewBox.baseVal.height / r.height : 1) : 1;
    const mx=(ev.clientX-r.left)*kx, my=(ev.clientY-r.top)*ky;
    let best=null, bd=Infinity;
    for(const q of proj){ const dx=q.cx-mx, dy=q.cy-my, d2=dx*dx+dy*dy;
      if(d2<bd){ bd=d2; best=q; } }
    return (best && bd<=R2) ? best.name : null;
  }
  // Re-render only when the species CHANGES: redrawing on every mousemove would fight
  // the pointer and throw away the tooltip mid-hover.
  svg.style.cursor="crosshair";
  svg.addEventListener("mousemove", ev=>{
    const sp=pick(ev);
    if(sp!==SEASON_HL){ SEASON_HL=sp; renderSeason(); }
  });
  // The card outlives every re-render, so wire its leave handler ONCE -- otherwise each
  // render stacks another listener and they all fire.
  if(!el.dataset.hoverWired){
    el.dataset.hoverWired="1";
    el.addEventListener("mouseleave", ()=>{ if(SEASON_HL!==null){ SEASON_HL=null; renderSeason(); } });
  }
  // The legend is the other natural place to aim at. Plot prefixes its class names with a
  // per-plot hash ("{hash}-swatch"), so match the SUFFIX -- and only the swatch itself,
  // never its "-swatches" container, whose class contains the same substring.
  el.querySelectorAll('[class*="swatch"]').forEach(sw=>{
    if(![...sw.classList].some(c=>/-swatch$/.test(c))) return;
    const name=(sw.textContent||"").trim();
    if(!name) return;
    sw.style.cursor="pointer";
    sw.addEventListener("mouseenter", ()=>{ if(SEASON_HL!==name){ SEASON_HL=name; renderSeason(); } });
  });
}

function renderTable(S){
  const g=aggBySpecies(S);
  const rows=Object.keys(g).map(name=>{ const rs=g[name];
    return {name, mornings:rs.length, n:rs.reduce((s,r)=>s+r.n,0),
      onset:median(rs.map(r=>r.onset)), offset:median(rs.map(r=>r.offset)),
      span:median(rs.map(r=>r.span)), peak:median(rs.map(r=>r.peak)), occ:median(rs.map(r=>r.occ))};
  }).sort((a,b)=> (a.onset==null)-(b.onset==null) || (a.onset-b.onset) || (b.n-a.n));
  const head=["Species","mornings","n","onset","offset","span","peak","occ"];
  const clickable = hasAudio() && S.length===1;
  const cell = d => clickable ? `<span class="listen" data-sp="${d.name}">${d.name}</span>` : d.name;
  const body=rows.map(d=>`<tr><td>${cell(d)}</td><td>${d.mornings}</td><td>${d.n}</td>`+
    `<td>${fmt(d.onset,0)}</td><td>${fmt(d.offset,0)}</td><td>${fmt(d.span,0)}</td>`+
    `<td>${fmt(d.peak,0)}</td><td>${fmt(d.occ,2)}</td></tr>`).join("");
  const tbl=document.getElementById("tbl");
  tbl.innerHTML = `<thead><tr>${head.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  if(clickable) tbl.querySelectorAll(".listen").forEach(s=> s.addEventListener("click", ()=>openAudioFor(s.dataset.sp)));
}

function renderFoot(){ document.getElementById("foot").innerHTML =
  `Onset/offset are the 5th/95th percentiles of detection times within the [dawn&minus;2h, dawn+4h] window; `+
  `mornings below the per-species detection floor get no onset. BirdNET does not separate song from call, so `+
  `"span" is vocal-activity span, not song-bout length. Charts by `+
  `<a href="https://observablehq.com/plot/" target="_blank" rel="noopener">Observable&nbsp;Plot</a> (library, vendored). `+
  `Regenerate with <code>tools/build_site.py</code> as new mornings arrive.`; }

function chipRow(elId, cls, onChange){
  const chips=document.getElementById(elId);
  const order=meta.top_species.filter(s=>meta.species.includes(s));
  const on=new Set(order.slice(0,5));
  chips.innerHTML=order.map(s=>`<label class="chip${on.has(s)?" on":""}"><input class="${cls}" type="checkbox" value="${s}"${on.has(s)?" checked":""}>${s}</label>`).join("");
  chips.addEventListener("change", e=>{ e.target.closest(".chip").classList.toggle("on", e.target.checked); onChange(); });
}
function buildChips(){
  chipRow("ecdfChips", "ecdf-sp", ()=>renderEcdf(scopedDays()));
  chipRow("seasonChips", "season-sp", renderSeason);
}

// Each chapter states what the data actually says, computed from the scoped period --
// a narrative that ignored the numbers would be decoration. Empty text hides itself, so
// a chapter with nothing to report says nothing rather than padding.
function renderFindings(S){
  const put=(id,html)=>{ const el=document.getElementById("find-"+id); if(el) el.innerHTML=html||""; };
  const g=aggBySpecies(S);
  const rows=Object.keys(g).map(n=>{ const rs=g[n].filter(r=>r.onset!=null);
    return rs.length ? {n, onset:median(rs.map(r=>r.onset)), det:rs.reduce((s,r)=>s+r.n,0)} : null;
  }).filter(Boolean).sort((a,b)=>a.onset-b.onset);

  if(rows.length){
    const f=rows[0], l=rows[rows.length-1];
    put("morning", `${f.n} leads at ${xTitle(f.onset)}; ${l.n} is last at ${xTitle(l.onset)}. `+
      `${rows.length} species have a defined onset over ${S.length} morning${S.length===1?"":"s"}.`);
    put("shape", `${rows.length} species charted. Curves that rise early and steeply are the ones `+
      `doing most of their singing before the light arrives.`);
  } else { put("morning",""); put("shape",""); }

  // Season and weather ignore the scope, so report from the whole record.
  const withOnset=summary.filter(r=>r.onset!=null);
  const days=new Set(withOnset.map(r=>r.date));
  put("season", days.size>1
    ? `${days.size} mornings from ${[...days].sort()[0]} to ${[...days].sort().pop()}. `+
      `A species needs 4 mornings before a trend line is drawn.`
    : "One morning so far — a trend needs several.");

  const wx=DATA.weather||{}, wd=Object.keys(wx).filter(d=>wx[d] && wx[d].t!=null);
  put("weather", wd.length
    ? `Weather for ${wd.length} morning${wd.length===1?"":"s"}. Treat pooled patterns carefully: `+
      `warm mornings also arrive later in the season.`
    : "No weather attached yet — it fills in when mornings are generated with a connection.");
  put("species", rows.length ? `Median values across ${S.length} morning${S.length===1?"":"s"} in scope.` : "");
}

function renderAll(){ refreshColors(); const S=scopedDays(); updateScopeLabel(S);
  setXOffset(S);
  renderTimeline(S); renderEcdf(S); renderHeat(S); renderSeason();
  renderWeather(S,"t","chart-temp","temperature at dawn (°C)","temp");
  renderWeather(S,"r","chart-rain","rain over the window (mm)","rain");
  renderTable(S); updateAudioCard(S); renderFindings(S); }

// ---- card collapsing + settings ---------------------------------------------------------
// Cards render into a hidden element as 0-wide, so anything opened has to be redrawn.
document.addEventListener("click", e=>{
  const h=e.target.closest(".card > h2"); if(!h) return;
  const card=h.parentElement, key=card.dataset.card; if(!key) return;
  const nowCollapsed=!card.classList.contains("collapsed");
  setCollapsed(key, nowCollapsed, true);
  const why=h.querySelector(".why"); if(why && !nowCollapsed) why.remove();
  if(!nowCollapsed) renderAll();
});
Object.entries(CARDS).forEach(([k,v])=>setCollapsed(k, v, false));

// Clock time needs civil dawn per morning. Payloads generated before that existed have no
// `dawn` map, so offer the option only when it can actually be honoured -- silently
// falling back to dawn-relative would look like the toggle was broken.
const CAN_CLOCK = Object.keys(DAWN).length > 0;

function setXMode(mode){
  if(mode==="clock" && !CAN_CLOCK) return;
  SETTINGS.xmode = mode; saveSettings(); applySettings(); renderAll();
}

// The toolbar control and the Settings dialog drive the SAME state, so both are repainted
// from SETTINGS rather than from whichever one was clicked.
function applySettings(){
  document.querySelectorAll('input[name="xmode"]').forEach(r=>{
    r.checked = r.value===SETTINGS.xmode;
    if(r.value==="clock" && !CAN_CLOCK) r.disabled = true;
  });
  document.querySelectorAll('#xmodeSeg button').forEach(b=>{
    b.setAttribute("aria-pressed", String(b.dataset.xmode===SETTINGS.xmode));
    if(b.dataset.xmode==="clock" && !CAN_CLOCK){
      b.disabled = true;
      b.title = "Needs civil-dawn times — regenerate this page to enable clock time";
    }
  });
  const ac=document.getElementById("autoCollapse"); if(ac) ac.checked = !!SETTINGS.autoCollapse;
}
const seg=document.getElementById("xmodeSeg");
if(seg) seg.addEventListener("click", e=>{
  const b=e.target.closest("button[data-xmode]");
  if(b && !b.disabled) setXMode(b.dataset.xmode);
});
document.querySelectorAll('input[name="xmode"]').forEach(r=> r.addEventListener("change", ()=>{
  if(r.checked) setXMode(r.value);
}));
document.getElementById("autoCollapse").addEventListener("change", e=>{
  SETTINGS.autoCollapse=e.target.checked; saveSettings();
  if(!SETTINGS.autoCollapse)                      // turning it off re-opens what it hid
    document.querySelectorAll(".card[data-card]").forEach(c=>{
      if(!(c.dataset.card in CARDS)) c.classList.remove("collapsed"); });
  renderAll();
});
document.getElementById("expandAll").addEventListener("click", ()=>{
  document.querySelectorAll(".card[data-card]").forEach(c=>setCollapsed(c.dataset.card, false, true));
  renderAll(); });
document.getElementById("collapseAll").addEventListener("click", ()=>{
  document.querySelectorAll(".card[data-card]").forEach(c=>setCollapsed(c.dataset.card, true, true)); });
applySettings();

aggSel.onchange = rebuildPeriods;
slider.oninput = renderAll;
document.getElementById("playBtn").onclick = playClip;
document.getElementById("prevDet").onclick = ()=>{ if(cur && cur.list && cur.pos>0) loadAt(cur.day, cur.list, cur.pos-1); };
document.getElementById("nextDet").onclick = ()=>{ if(cur && cur.list && cur.pos<cur.list.length-1) loadAt(cur.day, cur.list, cur.pos+1); };
(function initAudioSettings(){
  const ub=document.getElementById("urlBase"), sm=document.getElementById("specMode"),
        dp=document.getElementById("dirPick"), db=document.getElementById("dirBtn"), st=document.getElementById("dirStatus");
  if(!ub) return;
  ub.value=ABASE; sm.value=specMode; st.textContent="using served path";
  db.onclick=()=>dp.click();
  dp.onchange=ev=>{ let n=0; for(const f of ev.target.files){ if(f.name.toLowerCase().endsWith(".wav")){ audioFiles[f.name]=f; n++; } }
    for(const k in headerCache) delete headerCache[k];
    st.textContent = n ? `${n} local recording${n>1?"s":""}` : "no .wav found in folder"; reloadCur(); };
  ub.onchange=()=>{ ABASE=ub.value.trim()||meta.audio_base||"../data"; localStorage.setItem("dc_audio_base",ABASE);
    for(const k in headerCache) delete headerCache[k]; reloadCur(); };
  sm.onchange=()=>{ specMode=sm.value; localStorage.setItem("dc_spec_mode",specMode); renderClip(); };
})();
document.getElementById("theme").onclick = ()=>{ const curTheme=root.getAttribute("data-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"); root.setAttribute("data-theme", curTheme==="dark"?"light":"dark"); renderAll(); };
(function(){
  function wire(modalId, btnId){ const m=document.getElementById(modalId); if(!m) return null;
    const b=document.getElementById(btnId); if(b) b.onclick=()=>{ m.hidden=false; };
    const c=m.querySelector(".close"); if(c) c.onclick=()=>{ m.hidden=true; };
    m.addEventListener("click", e=>{ if(e.target===m) m.hidden=true; });   // backdrop closes
    return m; }
  const modals=[wire("guide","guideBtn"), wire("addModal","addBtn"),
                wire("setModal","setBtn")].filter(Boolean);
  addEventListener("keydown", e=>{ if(e.key==="Escape") modals.forEach(m=>m.hidden=true); });
  document.querySelectorAll("#addModal pre").forEach(pre=>{           // copy-to-clipboard on commands
    const b=document.createElement("button"); b.type="button"; b.className="copybtn"; b.textContent="Copy";
    b.onclick=()=>{ if(navigator.clipboard) navigator.clipboard.writeText(pre.textContent.trim());
      b.textContent="Copied ✓"; setTimeout(()=>{ b.textContent="Copy"; }, 1200); };
    pre.parentNode.insertBefore(b, pre.nextSibling);
  });
})();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>{ if(!root.hasAttribute("data-theme")) renderAll(); });
let rz; addEventListener("resize", ()=>{ clearTimeout(rz); rz=setTimeout(()=>{ renderAll(); renderClip(); }, 180); });

renderSubline(); renderTiles(); renderFoot(); buildChips(); rebuildPeriods();
}
// Static build embeds JSON in #data; the viewer leaves it empty and provides __bootFetch.
(function(){ const el=document.getElementById("data"), t=el?el.textContent.trim():"";
  if(t){ boot(JSON.parse(t)); } else if(window.__bootFetch){ window.__bootFetch(boot); } })();

// Link back to the local control panel -- but ONLY when one is really there.
// This same template builds the PUBLIC site, so the link must never appear for visitors:
// we bail unless the page is on loopback, and then only add it if /api/state answers.
// (The loopback test comes first so the public site issues no request at all.)
(function(){
  const local = ["127.0.0.1","localhost","[::1]"].includes(location.hostname);
  if(!local) return;
  const ctl = new AbortController(); setTimeout(()=>ctl.abort(), 1500);
  fetch("/api/state", {signal: ctl.signal})
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(() => {
      const hb = document.querySelector(".hbtns"); if(!hb) return;
      const a = document.createElement("a");
      a.href = "/"; a.title = "Run models, rebuild, publish";
      a.innerHTML = '<button class="theme">⚙ Control panel</button>';
      hb.appendChild(a);
    })
    .catch(()=>{});          // served by serve.py or opened as a file: no panel, no link
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
