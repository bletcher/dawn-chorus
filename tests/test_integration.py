"""End-to-end: build a synthetic station with a KNOWN planted signal and confirm
the whole pipeline (schema load -> solar anchor -> quantile onset -> weather join
-> regression) recovers it. This is the regression test that pins correctness.

The generator bakes in `onset_shift = -1.4 * temperature_anomaly`, i.e. warmer
mornings -> earlier singing, so every species' onset~temperature slope should be
negative. It also assigns fixed per-species onsets, so ordering is recoverable.
"""
from __future__ import annotations

import random

import dawnchorus as dc

# Provided on sys.path by tests/conftest.py (tools/ is a script dir, not a package).
import make_synthetic_db as gen


def _build(tmp_path):
    db = tmp_path / "demo.db"
    wx = tmp_path / "demo_weather.csv"
    random.seed(7)  # deterministic DB + mock weather, independent of import order
    gen.make(str(db), str(wx))
    return db, wx


def test_pipeline_recovers_planted_temperature_signal(tmp_path):
    db, wx = _build(tmp_path)
    out = dc.run(str(db), gen.LAT, gen.LON, gen.TZ, min_confidence=0.6,
                 weather=True, weather_cache=str(wx))

    ms = out["morning_summary"]
    assert not ms.empty
    assert "temperature_2m" in ms.columns  # weather actually joined

    wr = out["weather_response"]
    temp = wr[wr["covariate"] == "temperature_2m"]
    assert len(temp) >= 6                       # a fit for most species
    # Direction of the planted effect is recovered: warmer -> earlier onset.
    assert temp["slope"].mean() < 0
    assert (temp["slope"] < 0).mean() >= 0.75   # the large majority are negative


def test_pipeline_recovers_species_onset_ordering(tmp_path):
    db, wx = _build(tmp_path)
    out = dc.run(str(db), gen.LAT, gen.LON, gen.TZ, min_confidence=0.6,
                 weather=True, weather_cache=str(wx))

    eq = out["ecdf_quantiles_month"]
    m5 = eq[(eq["quantile"] == 0.05) & (eq["month"] == 5)]
    robin = m5.loc[m5["common_name"] == "American Robin", "t_min"].iloc[0]
    chickadee = m5.loc[m5["common_name"] == "Black-capped Chickadee", "t_min"].iloc[0]

    # Generator: robin onset0 = -40 (before dawn), chickadee = +10 (after).
    assert robin < 0
    assert robin < chickadee
