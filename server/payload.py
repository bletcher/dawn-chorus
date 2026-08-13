"""
Build the dashboard payload (the JSON the frontend renders) from a detections DataFrame.

This is the server-side analogue of tools/build_site.build_data, but sourced from the
database instead of BirdNET result files, and without the audio pieces (recordings stay on
contributors' machines, so a hosted viewer gets charts + labels, not click-to-listen).

Reuses the tested `dawnchorus` library (solar anchoring, quantile onset). The frontend does
the time-scope aggregation and ECDF client-side from these per-morning building blocks.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import dawnchorus as dc
from dawnchorus import morning_summary
from dawnchorus.phenology import DEFAULTS, _anchor_col


def _recorder_meta(det: pd.DataFrame) -> list[dict]:
    """Which recorder(s) produced these detections, and over what span.

    Empty when the source predates recorder tagging — the viewer treats that as one
    unnamed deployment, so old payloads keep rendering unchanged.
    """
    if "recorder" not in det.columns:
        return []
    out = []
    for rid, g in det.dropna(subset=["recorder"]).groupby("recorder"):
        dates = sorted(str(d) for d in pd.unique(g["date"]))
        out.append({"id": str(rid), "n": int(len(g)),
                    "first": dates[0] if dates else None,
                    "last": dates[-1] if dates else None})
    return sorted(out, key=lambda r: r["id"])


def build_payload(det_all: pd.DataFrame, lat: float, lon: float, tz: str,
                  min_conf: float = 0.5, label_min_conf: float = 0.25,
                  weather: bool = True, weather_cache: str | None = None,
                  weather_source: str = "archive") -> dict:
    """det_all: columns datetime (naive/aware local), scientific_name, common_name, confidence.

    An optional `recorder` column is carried into `meta.recorders` so a payload always
    states which hardware produced it. Detectability is hardware-dependent, so a site's
    numbers are only comparable across time while that list is unchanged.
    """
    cfg = DEFAULTS
    acol = _anchor_col(cfg["anchor"])                       # min_from_dawn
    lo, hi, bw = cfg["window_start_min"], cfg["window_end_min"], cfg["bin_min"]

    model = dc.SolarModel(lat, lon, tz)
    ann = model.annotate(det_all)
    ann["date"] = pd.to_datetime(ann["datetime"]).dt.date

    # --- charts: detections at >= min_conf ---
    det = ann[ann["confidence"] >= min_conf].copy()
    ms = morning_summary(det, None)                          # per-morning onset/offset/occupancy

    edges = np.arange(lo, hi + bw, bw)
    nbins = len(edges) - 1
    grid = [round(float(edges[i]) + bw / 2.0, 1) for i in range(nbins)]
    win = det[(det[acol] >= lo) & (det[acol] < hi)].copy()
    win["bin"] = pd.cut(win[acol], edges, labels=False)

    totals = win.groupby("common_name").size().sort_values(ascending=False)
    heat_set = {s for s in totals.index if totals[s] >= 5}
    mean_t = win.groupby("common_name")[acol].mean().sort_values()
    species = [s for s in mean_t.index if s in heat_set]
    sp_idx = {s: i for i, s in enumerate(species)}

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

    s = ms.rename(columns={"scientific_name": "sci", "common_name": "name", "n_detections": "n",
                           "onset_min": "onset", "offset_min": "offset", "span_min": "span",
                           "peak_min": "peak", "occupancy": "occ"})
    s["date"] = s["date"].astype(str)
    s = s.round({"onset": 1, "offset": 1, "span": 1, "peak": 1, "occ": 2})
    summary = json.loads(s[["date", "name", "sci", "n", "onset", "offset", "span",
                            "peak", "occ"]].to_json(orient="records"))

    # --- labels: detections at >= label_min_conf (lower, so more calls show) ---
    lab = ann[ann["confidence"] >= label_min_conf]
    # Labels feed the spectrogram browser, not the metrics, so they are NOT clipped to
    # the analysis window -- otherwise the tail of every recording browses unlabelled.
    winl = lab
    label_species = sorted(winl["common_name"].unique().tolist())
    lsp = {s: i for i, s in enumerate(label_species)}
    dets_by_day = {}
    for d, g in winl.groupby("date"):
        dets_by_day[str(d)] = [[round(float(a), 2), lsp[sp], round(float(c), 2)]
                               for sp, a, c in zip(g["common_name"], g[acol], g["confidence"])]

    valid = ms.dropna(subset=["onset_min"])
    earliest = None
    if not valid.empty:
        r = valid.loc[valid["onset_min"].idxmin()]
        earliest = {"name": r["common_name"], "onset": round(float(r["onset_min"]), 1)}

    # --- per-morning weather (Open-Meteo): temperature at dawn + rain over the window ---
    wx_by_day = {}
    if weather:
        try:
            from dawnchorus import weather as _wx
            mdays = sorted(pd.unique(ms["date"]))          # NB: don't shadow `days` (string dates in meta)
            hourly = _wx.fetch_hourly(lat, lon, min(mdays), max(mdays), tz,
                                      cache_path=weather_cache, source=weather_source)
            mw = _wx.morning_weather(hourly, model, mdays)
            for r in mw.itertuples():
                t, rain = getattr(r, "temp_at_anchor", np.nan), getattr(r, "precip_sum", np.nan)
                wx_by_day[str(r.date)] = {"t": None if pd.isna(t) else round(float(t), 1),
                                          "r": None if pd.isna(rain) else round(float(rain), 2)}
        except Exception:
            wx_by_day = {}     # no network / no coverage -> charts show "weather not available yet"

    # Civil dawn (seconds past local midnight) per morning. Everything is stored relative
    # to dawn, so this is what lets a viewer offer a clock-time axis: dawn drifts through
    # the season, and a fixed offset would misplace every point by up to an hour.
    dawn_by_day = {}
    for d in sorted(pd.unique(ms["date"])):
        dw = model.dawn(d)
        if dw is not None:
            dawn_by_day[str(d)] = dw.hour * 3600 + dw.minute * 60 + dw.second

    meta = {
        "mornings": sorted(str(d) for d in pd.unique(ms["date"])),
        "days": days,
        "n_species": int(det["scientific_name"].nunique()),
        "n_detections": int(len(det)),
        "min_confidence": min_conf,
        "lat": lat, "lon": lon, "tz": tz,
        "window": [lo, hi], "bin": bw, "grid": grid,
        "species": species,
        "top_species": [s for s in totals.index if s in heat_set],
        "earliest": earliest,
        "label_species": label_species,
        "recorders": _recorder_meta(det),
    }
    return {"meta": meta, "summary": summary, "counts": counts, "day_keys": day_keys,
            "dets": dets_by_day, "weather": wx_by_day, "dawn": dawn_by_day}
