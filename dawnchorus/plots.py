"""
Reference plots. Deliberately minimal (matplotlib only) -- the package's main
deliverable is tidy tables you can pipe into Observable / ggplot2 for polished
figures. These exist so `dawnchorus` produces something visual out of the box.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def onset_by_species(species_pheno: pd.DataFrame, period, anchor: str = "sunrise", ax=None):
    """Horizontal onset ranges (IQR) per species for one period, ordered earliest first."""
    d = species_pheno[species_pheno[species_pheno.columns[0]] == period].copy()
    d = d.sort_values("onset_median")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.hlines(y, d["onset_q25"], d["onset_q75"], color="#6b8e9e", lw=4, alpha=.7)
    ax.plot(d["onset_median"], y, "o", color="#22303a")
    ax.axvline(0, color="#c25b3a", ls="--", lw=1, label=f"{anchor}")
    ax.set_yticks(y); ax.set_yticklabels(d["common_name"])
    ax.set_xlabel(f"minutes from {anchor}  (median, IQR)")
    ax.set_title(f"Song onset by species — period {period}")
    ax.legend(loc="lower right", fontsize=8)
    ax.margins(y=0.02)
    return ax


def occupancy_heatmap(det: pd.DataFrame, config: dict, ax=None):
    """Species x solar-minute occupancy averaged across all mornings."""
    from .phenology import _anchor_col
    acol = _anchor_col(config.get("anchor", "sunrise"))
    lo, hi, bw = config["window_start_min"], config["window_end_min"], config["bin_min"]
    win = det[(det[acol] >= lo) & (det[acol] < hi)].copy()

    edges = np.arange(lo, hi + bw, bw)
    win["bin"] = pd.cut(win[acol], edges, labels=edges[:-1])
    n_mornings = win["date"].nunique()
    # fraction of mornings each species is present in each bin
    grid = (win.groupby(["common_name", "bin"], observed=True)["date"].nunique()
            .unstack(fill_value=0) / max(1, n_mornings))
    order = grid.mul(grid.columns.astype(float), axis=1).sum(axis=1) / grid.sum(axis=1).replace(0, np.nan)
    grid = grid.loc[order.sort_values().index]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(grid))))
    im = ax.imshow(grid.values, aspect="auto", cmap="magma",
                   extent=[lo, hi, len(grid), 0], vmin=0, vmax=1)
    ax.axvline(0, color="w", ls="--", lw=1)
    ax.set_yticks(np.arange(len(grid)) + .5); ax.set_yticklabels(grid.index)
    ax.set_xlabel(f"minutes from {config.get('anchor','sunrise')}")
    ax.set_title(f"Mean vocal occupancy ({n_mornings} mornings)")
    plt.colorbar(im, ax=ax, label="fraction of mornings present")
    return ax


def composition_area(comp: pd.DataFrame, ax=None):
    """Stacked share of the chorus by period."""
    wide = comp.pivot_table(index="period", columns="common_name",
                            values="share", fill_value=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(wide.index, wide.T.values, labels=wide.columns)
    ax.set_xlabel("period"); ax.set_ylabel("share of dawn-window detections")
    ax.set_title("Seasonal composition of the dawn chorus")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8, ncol=1)
    return ax


def onset_vs_temperature(summary_wx, anchor="dawn", max_species=6, ax=None):
    """Scatter + OLS fit of onset vs window-mean temperature, top species by n."""
    if "temperature_2m" not in summary_wx:
        return None
    d = summary_wx.dropna(subset=["onset_min", "temperature_2m"])
    top = d["common_name"].value_counts().head(max_species).index
    d = d[d["common_name"].isin(top)]
    if d.empty:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, (sp, g) in enumerate(d.groupby("common_name")):
        c = cmap(i % 10)
        ax.scatter(g["temperature_2m"], g["onset_min"], s=12, alpha=.5, color=c, label=sp)
        if g["temperature_2m"].nunique() >= 3:
            x = np.sort(g["temperature_2m"].to_numpy(float))
            m, b = np.polyfit(g["temperature_2m"].to_numpy(float),
                              g["onset_min"].to_numpy(float), 1)
            ax.plot(x, m * x + b, color=c, lw=2)
    ax.axhline(0, color="#c25b3a", ls="--", lw=1)
    ax.set_xlabel("window-mean temperature (°C)")
    ax.set_ylabel(f"onset (minutes from {anchor})")
    ax.set_title("Song onset vs morning temperature")
    ax.legend(fontsize=8, loc="best")
    return ax


def ecdf_small_multiples(ecdf_df, by="month", anchor="dawn", max_species=8,
                         show_band=True, band_for=None):
    """One panel per species; each period a coloured mean-ECDF curve.

    Horizontal guides at 0.05 / 0.5 / 0.95 make onset / median / offset readable;
    the vertical line marks the anchor. Optionally shade the p25-p75 band for one
    reference period (band_for) to show morning-to-morning spread without clutter.
    """
    species = (ecdf_df.groupby("common_name")["n_detections"].sum()
               .sort_values(ascending=False).head(max_species).index.tolist())
    n = len(species)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.4 * nrow),
                             squeeze=False, sharex=True)
    periods = sorted(ecdf_df[by].unique())
    cmap = plt.get_cmap("viridis")
    cnorm = {p: cmap(i / max(1, len(periods) - 1)) for i, p in enumerate(periods)}

    for k, sp in enumerate(species):
        ax = axes[k // ncol][k % ncol]
        d = ecdf_df[ecdf_df["common_name"] == sp]
        for p in periods:
            g = d[d[by] == p].sort_values("t_min")
            if g.empty:
                continue
            ax.plot(g["t_min"], g["F"], color=cnorm[p], lw=1.6, label=str(p))
            if show_band and (band_for is not None) and p == band_for:
                ax.fill_between(g["t_min"], g["F_p25"], g["F_p75"],
                                color=cnorm[p], alpha=.2)
        for q in (0.05, 0.5, 0.95):
            ax.axhline(q, color="0.8", lw=.6, zorder=0)
        ax.axvline(0, color="#c25b3a", ls="--", lw=1)
        ax.set_title(sp, fontsize=9); ax.set_ylim(0, 1)
        ax.set_ylabel("F (cum. share)", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel(f"minutes from {anchor}", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title=by, loc="center left",
               bbox_to_anchor=(1.0, .5), fontsize=8)
    fig.suptitle("Empirical CDF of vocal activity by " + by, y=1.0, fontsize=11)
    fig.tight_layout()
    return fig
