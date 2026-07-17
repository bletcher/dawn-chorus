"""
Empirical cumulative distribution of vocal activity over solar time.

This is the robust primitive the scalar onset/offset metrics are only summaries of.
For a species on a morning, F(t) = fraction of that species' detections that have
occurred by solar-relative minute t. Adding one spurious detection moves F by just
1/n (a single negligible step), so the curve -- and every quantile read from it --
is insensitive to outliers by construction.

The sampling unit is the MORNING, not the detection (detections within a morning are
pseudoreplicated). So we compute each morning's ECDF on a common grid, then average
across mornings and report a pointwise band. The result reads as: "on a typical
morning in this stratum, what fraction of the species' vocal activity has happened by
time t?" -- with morning-to-morning spread shown by the band.

Outputs
-------
species_ecdf(...)   tidy grid table: by, species, t_min, F (mean), F_p25, F_p75, n
ecdf_quantiles(...) onset/median/offset read as crossings of the mean ECDF
ecdf_distance(...)  KS statistic, 1-D Wasserstein (minutes), median shift between two
                    strata for a species -- dependency-free, descriptive (do formal
                    tests in R / scipy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .phenology import _anchor_col, DEFAULTS


def _grid(cfg):
    return np.arange(cfg["window_start_min"], cfg["window_end_min"] + cfg["bin_min"], cfg["bin_min"])


def _ecdf_on_grid(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """F(t)=fraction of `times` <= each grid point. 0 before first, 1 at/after last."""
    ts = np.sort(times)
    return np.searchsorted(ts, grid, side="right") / len(ts)


def _period(dates: pd.Series, by: str) -> pd.Series:
    ts = pd.to_datetime(dates)
    if by == "month":
        return ts.dt.month
    if by == "week":
        return ts.dt.isocalendar().week.astype(int)
    if by == "doy":
        return ts.dt.dayofyear
    raise ValueError("by must be 'month', 'week', or 'doy'")


def species_ecdf(det: pd.DataFrame, by: str = "month", config: dict | None = None,
                 min_detections: int | None = None, pooled: bool = False) -> pd.DataFrame:
    """Grid-evaluated ECDF of vocal activity per (period, species).

    pooled=False (default): mean of per-morning ECDFs (+ p25/p75 band), morning as
    the replicate. pooled=True: one ECDF over all detections in the stratum (simpler,
    but pseudoreplicates within morning).
    """
    cfg = {**DEFAULTS, **(config or {})}
    acol = _anchor_col(cfg["anchor"])
    floor = cfg["min_detections_per_morning"] if min_detections is None else min_detections
    grid = _grid(cfg)

    win = det[(det[acol] >= cfg["window_start_min"]) & (det[acol] < cfg["window_end_min"])].copy()
    win["period"] = _period(win["date"], by)

    rows = []
    for (per, sp), g in win.groupby(["period", "scientific_name"], sort=True):
        common = g["common_name"].iloc[0]
        if pooled:
            F = _ecdf_on_grid(g[acol].to_numpy(), grid)
            lo = hi = np.full_like(F, np.nan)
            n_mornings = g["date"].nunique()
        else:
            curves = []
            for _, gm in g.groupby("date"):
                if len(gm) >= floor:
                    curves.append(_ecdf_on_grid(gm[acol].to_numpy(), grid))
            if not curves:
                continue
            M = np.vstack(curves)
            F = M.mean(axis=0)
            lo = np.quantile(M, 0.25, axis=0)
            hi = np.quantile(M, 0.75, axis=0)
            n_mornings = len(curves)
        for t, f, l, h in zip(grid, F, lo, hi):
            rows.append(dict(period=per, scientific_name=sp, common_name=common,
                             t_min=float(t), F=float(f), F_p25=float(l), F_p75=float(h),
                             n_mornings=int(n_mornings), n_detections=int(len(g))))
    out = pd.DataFrame(rows)
    return out.rename(columns={"period": by})


def _crossing(grid: np.ndarray, F: np.ndarray, q: float) -> float:
    """Solar minute where a monotone F first reaches q (linear-interpolated)."""
    idx = int(np.searchsorted(F, q, side="left"))
    if idx <= 0:
        return float(grid[0])
    if idx >= len(F):
        return float(grid[-1])
    f0, f1, t0, t1 = F[idx - 1], F[idx], grid[idx - 1], grid[idx]
    if f1 == f0:
        return float(t0)
    return float(t0 + (q - f0) * (t1 - t0) / (f1 - f0))


def ecdf_quantiles(ecdf_df: pd.DataFrame, by: str = "month",
                   quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Read onset/median/offset (and any quantiles) off the mean ECDF curves."""
    rows = []
    for (per, sp), g in ecdf_df.groupby([by, "scientific_name"]):
        g = g.sort_values("t_min")
        grid, F = g["t_min"].to_numpy(), g["F"].to_numpy()
        for q in quantiles:
            rows.append(dict(**{by: per}, scientific_name=sp,
                             common_name=g["common_name"].iloc[0],
                             quantile=q, t_min=_crossing(grid, F, q),
                             n_mornings=int(g["n_mornings"].iloc[0])))
    return pd.DataFrame(rows)


# --- distributional distances (dependency-free, descriptive) -----------------

def _ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, allv, "right") / len(a)
    cb = np.searchsorted(b, allv, "right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _ks_pvalue(D, n1, n2):
    en = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(en) + 0.12 + 0.11 / np.sqrt(en)) * D
    k = np.arange(1, 101)
    q = 2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2))
    return float(min(max(q, 0.0), 1.0))


def _wasserstein_1d(a, b):
    """W1 = integral |Fa - Fb| dt, in the same units as the samples (minutes)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    d = np.diff(allv)
    ca = np.searchsorted(a, allv[:-1], "right") / len(a)
    cb = np.searchsorted(b, allv[:-1], "right") / len(b)
    return float(np.sum(np.abs(ca - cb) * d))


def ecdf_distance(det: pd.DataFrame, scientific_name: str, by: str,
                  group_a, group_b, config: dict | None = None) -> dict:
    """Compare one species' timing distribution between two strata.

    Pooled over detections (fast, descriptive). NOTE: this pseudoreplicates within
    morning -- treat KS/Wasserstein as effect-size descriptors, not hypothesis tests;
    for inference compare per-morning summaries with morning as the unit, in R.
    """
    cfg = {**DEFAULTS, **(config or {})}
    acol = _anchor_col(cfg["anchor"])
    win = det[(det[acol] >= cfg["window_start_min"]) & (det[acol] < cfg["window_end_min"])].copy()
    win = win[win["scientific_name"] == scientific_name]
    win["period"] = _period(win["date"], by)
    a = win[win["period"] == group_a][acol].to_numpy()
    b = win[win["period"] == group_b][acol].to_numpy()
    if len(a) < 5 or len(b) < 5:
        return dict(scientific_name=scientific_name, group_a=group_a, group_b=group_b,
                    n_a=len(a), n_b=len(b), ks=np.nan, ks_p=np.nan,
                    wasserstein_min=np.nan, median_shift_min=np.nan)
    D = _ks_stat(a, b)
    return dict(scientific_name=scientific_name, group_a=group_a, group_b=group_b,
                n_a=len(a), n_b=len(b), ks=D, ks_p=_ks_pvalue(D, len(a), len(b)),
                wasserstein_min=_wasserstein_1d(a, b),
                median_shift_min=float(np.median(b) - np.median(a)))
