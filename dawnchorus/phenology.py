"""
Per-species, per-morning vocal phenology.

Given solar-anchored detections restricted to a fixed morning window, we summarise
each (species, morning) with:

    onset_min   : robust start of vocal activity (a low quantile of detection times)
    offset_min  : robust end   of vocal activity (the matching high quantile)
    span_min    : offset - onset  ("how long they sing", operationally)
    peak_min    : solar-relative minute of the busiest time bin
    occupancy   : fraction of bins between onset and offset that hold >=1 detection
    n_detections: raw count in the morning window (above threshold)

DESIGN (why quantiles, not a count rule):
  Onset/offset are the `onset_quantile` and `1 - onset_quantile` quantiles of the
  detection times within the window (default 0.05 / 0.95). A single stray daytime
  false positive barely moves a 5th percentile when real singing is present, so
  this is robust WITHOUT a detection-count threshold -- and unlike a "k detections
  in a window" rule it is symmetric, so it does not systematically push onset later
  for quiet species. Set onset_quantile: 0.0 for literal first/last detection.

  Reliability floor: mornings with fewer than `min_detections_per_morning`
  detections get NaN onset/offset (too few points to place a quantile), while still
  contributing their raw count and presence to composition/richness.

METHODOLOGICAL NOTES (read before trusting a number):
  * BirdNET emits ~one detection per 3-second window per species and does NOT
    distinguish song from call. So span_min is a *vocal-activity* span, not a
    literal song-bout length, and occupancy is a proxy for the fraction of the
    morning a species was detectably vocalising.
  * Detectability depends on distance, wind, and mic directionality. Keep the
    anchor and hardware fixed across the season for comparisons to hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = dict(
    anchor="dawn",                 # "dawn" (civil twilight begin) or "sunrise"
    window_start_min=-120.0,       # window opens 2 h before the anchor
    window_end_min=240.0,          # and closes 4 h after
    bin_min=5.0,                   # time-bin width (min) for occupancy / peak
    onset_quantile=0.05,           # onset = 5th pct of detection times; offset = 95th
    min_detections_per_morning=5,  # below this, onset/offset undefined (NaN)
)


def _anchor_col(anchor: str) -> str:
    if anchor == "sunrise":
        return "min_from_sunrise"
    if anchor == "dawn":
        return "min_from_dawn"
    raise ValueError("anchor must be 'sunrise' or 'dawn'")


def morning_summary(det: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """One row per (date, species) with the phenology metrics above."""
    cfg = {**DEFAULTS, **(config or {})}
    acol = _anchor_col(cfg["anchor"])
    lo, hi, bin_min = cfg["window_start_min"], cfg["window_end_min"], cfg["bin_min"]
    q = cfg["onset_quantile"]
    min_n = cfg["min_detections_per_morning"]
    edges = np.arange(lo, hi + bin_min, bin_min)
    centers = edges[:-1] + bin_min / 2.0

    win = det[(det[acol] >= lo) & (det[acol] < hi)].copy()
    rows = []
    for (d, sp), g in win.groupby(["date", "scientific_name"], sort=False):
        m = np.sort(g[acol].to_numpy())
        n = len(m)
        counts, _ = np.histogram(m, bins=edges)

        if n >= min_n:
            onset = float(np.quantile(m, q))
            offset = float(np.quantile(m, 1.0 - q))
            span = offset - onset if offset > onset else np.nan
            in_span = (centers >= onset) & (centers <= offset)
            nb = int(in_span.sum())
            occ = float((counts[in_span] > 0).sum()) / nb if nb else np.nan
        else:
            onset = offset = span = occ = np.nan

        peak_min = float(centers[int(np.argmax(counts))]) if counts.sum() else np.nan
        rows.append(dict(
            date=d, scientific_name=sp, common_name=g["common_name"].iloc[0],
            n_detections=int(n), onset_min=onset, offset_min=offset, span_min=span,
            peak_min=peak_min, occupancy=occ,
            raw_first_min=float(m.min()), raw_last_min=float(m.max()),
            month=pd.Timestamp(d).month, doy=pd.Timestamp(d).dayofyear,
        ))
    cols = ["date", "month", "doy", "scientific_name", "common_name", "n_detections",
            "onset_min", "offset_min", "span_min", "peak_min", "occupancy",
            "raw_first_min", "raw_last_min"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["date", "scientific_name"]).reset_index(drop=True))


def species_phenology(summary: pd.DataFrame, by: str = "month",
                      min_mornings: int = 3) -> pd.DataFrame:
    """Aggregate morning summaries into per-species onset/span distributions.

    `by` groups mornings (e.g. 'month'); onset/span are summarised as median + IQR
    across the mornings a species was present. Species with fewer than
    `min_mornings` qualifying mornings in a group are dropped as unreliable.
    """
    valid = summary.dropna(subset=["onset_min", "span_min"])

    def q(s, p):
        return s.quantile(p)

    agg = (valid.groupby([by, "scientific_name", "common_name"])
           .agg(mornings=("date", "nunique"),
                onset_median=("onset_min", "median"),
                onset_q25=("onset_min", lambda s: q(s, .25)),
                onset_q75=("onset_min", lambda s: q(s, .75)),
                span_median=("span_min", "median"),
                span_q25=("span_min", lambda s: q(s, .25)),
                span_q75=("span_min", lambda s: q(s, .75)),
                occupancy_median=("occupancy", "median"),
                detections_per_morning=("n_detections", "mean"))
           .reset_index())
    return agg[agg["mornings"] >= min_mornings].reset_index(drop=True)
