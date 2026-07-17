"""Phenology: quantile onset robustness and the min-detections reliability floor."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from dawnchorus import morning_summary


def _det(min_from_dawn, d=date(2025, 5, 15), sp="turdus_migratorius", cn="American Robin"):
    """Build a detections frame with a pre-computed solar-time column."""
    return pd.DataFrame({
        "min_from_dawn": np.asarray(min_from_dawn, dtype=float),
        "date": [d] * len(min_from_dawn),
        "scientific_name": [sp] * len(min_from_dawn),
        "common_name": [cn] * len(min_from_dawn),
    })


def test_onset_is_robust_to_a_stray_daytime_false_positive():
    real = np.linspace(0, 60, 20)          # genuine morning singing, 0..60 min
    base = morning_summary(_det(real))
    onset_base = base["onset_min"].iloc[0]

    with_fp = morning_summary(_det(np.append(real, 200.0)))  # one late FP in-window
    onset_fp = with_fp["onset_min"].iloc[0]

    # The 5th-percentile onset barely moves (< 5 min)...
    assert abs(onset_fp - onset_base) < 5
    # ...even though the literal last detection jumps out to the false positive.
    assert with_fp["raw_last_min"].iloc[0] == 200.0
    assert base["raw_last_min"].iloc[0] <= 60


def test_too_few_detections_gives_nan_onset_but_keeps_count():
    row = morning_summary(_det([10.0, 20.0, 30.0])).iloc[0]  # 3 < default floor of 5
    assert np.isnan(row["onset_min"])
    assert np.isnan(row["offset_min"])
    assert row["n_detections"] == 3


def test_span_is_offset_minus_onset():
    row = morning_summary(_det(np.linspace(-30, 90, 40))).iloc[0]
    assert row["span_min"] == row["offset_min"] - row["onset_min"]
    assert row["onset_min"] < row["offset_min"]


def test_detections_outside_window_are_excluded():
    # window default is [-120, 240); a detection at 500 min must not appear.
    summ = morning_summary(_det(np.append(np.linspace(0, 60, 10), 500.0)))
    assert summ["n_detections"].iloc[0] == 10
