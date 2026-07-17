"""
Seasonal turnover of the dawn chorus.

Two questions:
  1. How does each species' *share* of the chorus change across the season?
  2. How does chorus richness (species per morning) change across the season?

We restrict to the morning window and aggregate by a grouping period (month by
default, or ISO week for finer resolution). Shares are computed from detection
counts within the window, so a species that sings a lot weighs more than a
one-note visitor -- consistent with "species contributions" in the original
question. Richness is averaged per morning so uneven sampling effort across
months doesn't inflate a busy month.
"""

from __future__ import annotations

import pandas as pd

from .phenology import _anchor_col, DEFAULTS


def _windowed(det: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(config or {})}
    acol = _anchor_col(cfg["anchor"])
    return det[(det[acol] >= cfg["window_start_min"]) & (det[acol] < cfg["window_end_min"])].copy()


def _period(det: pd.DataFrame, by: str) -> pd.Series:
    ts = pd.to_datetime(det["date"])
    if by == "month":
        return ts.dt.month
    if by == "week":
        return ts.dt.isocalendar().week.astype(int)
    if by == "doy":
        return ts.dt.dayofyear
    raise ValueError("by must be 'month', 'week', or 'doy'")


def composition(det: pd.DataFrame, by: str = "month",
                config: dict | None = None, top_n: int | None = 12) -> pd.DataFrame:
    """Long-form share table: one row per (period, species) with its detection share.

    If top_n is set, species outside the season-wide top_n by total detections are
    collapsed into an 'Other' row so stacked plots stay legible.
    """
    win = _windowed(det, config)
    win = win.assign(period=_period(win, by))

    counts = (win.groupby(["period", "common_name"]).size()
              .rename("detections").reset_index())
    if top_n is not None:
        totals = counts.groupby("common_name")["detections"].sum().sort_values(ascending=False)
        keep = set(totals.head(top_n).index)
        counts["common_name"] = counts["common_name"].where(counts["common_name"].isin(keep), "Other")
        counts = counts.groupby(["period", "common_name"], as_index=False)["detections"].sum()

    period_tot = counts.groupby("period")["detections"].transform("sum")
    counts["share"] = counts["detections"] / period_tot
    return counts.sort_values(["period", "detections"], ascending=[True, False]).reset_index(drop=True)


def richness(det: pd.DataFrame, by: str = "month", config: dict | None = None) -> pd.DataFrame:
    """Per-period richness: mean species per morning, plus cumulative species pool."""
    win = _windowed(det, config)
    win = win.assign(period=_period(win, by))

    per_morning = (win.groupby(["period", "date"])["scientific_name"]
                   .nunique().rename("species_that_morning").reset_index())
    pool = (win.groupby("period")["scientific_name"].nunique()
            .rename("species_pool").reset_index())
    mean_morn = (per_morning.groupby("period")["species_that_morning"].mean()
                 .rename("mean_species_per_morning").reset_index())
    mornings = (per_morning.groupby("period")["date"].nunique()
                .rename("mornings_sampled").reset_index())

    out = pool.merge(mean_morn, on="period").merge(mornings, on="period")
    return out.sort_values("period").reset_index(drop=True)
