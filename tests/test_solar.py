"""Solar-time anchoring and the high-latitude (polar day/night) guard."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dawnchorus import SolarModel


def _frame(dt_str):
    return pd.DataFrame({"datetime": [pd.Timestamp(dt_str)],
                         "scientific_name": ["x"], "common_name": ["X"]})


def test_implausible_latlon_rejected():
    with pytest.raises(ValueError):
        SolarModel(999.0, 0.0, "UTC")


def test_dawn_precedes_sunrise_so_min_from_dawn_is_larger():
    solar = SolarModel(42.53, -72.53, "America/New_York")
    out = solar.annotate(_frame("2025-05-15 05:30:00"))
    # For a fixed detection, min_from_dawn - min_from_sunrise = sunrise - dawn > 0.
    assert out["min_from_dawn"].iloc[0] > out["min_from_sunrise"].iloc[0]
    gap = out["min_from_dawn"].iloc[0] - out["min_from_sunrise"].iloc[0]
    assert 15 < gap < 60  # civil twilight length at mid-latitude, minutes


def test_detection_at_dawn_is_near_zero_from_dawn():
    solar = SolarModel(42.53, -72.53, "America/New_York")
    d = date(2025, 5, 15)
    dawn_local = solar.dawn(d)
    out = solar.annotate(_frame(dawn_local.replace(tzinfo=None)))
    assert abs(out["min_from_dawn"].iloc[0]) < 1e-6


def test_polar_day_warns_and_yields_nan_not_crash():
    # Svalbard on the solstice: midnight sun -> no sunrise, no civil dawn.
    solar = SolarModel(78.22, 15.65, "Arctic/Longyearbyen")
    assert solar.dawn(date(2025, 6, 21)) is None
    assert solar.sunrise(date(2025, 6, 21)) is None
    with pytest.warns(UserWarning, match="polar day/night"):
        out = solar.annotate(_frame("2025-06-21 03:00:00"))
    assert np.isnan(out["min_from_dawn"].iloc[0])
    assert np.isnan(out["min_from_sunrise"].iloc[0])
