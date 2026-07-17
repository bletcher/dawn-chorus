"""
Weather covariates via Open-Meteo (no API key required).

Two things happen here:
  1. fetch_hourly()     -- pull hourly weather for the station location/date range
                           from Open-Meteo's ERA5 archive, with local caching.
  2. morning_weather()  -- reduce hourly weather to ONE row per morning: conditions
                           at the solar anchor plus window means/sums. These are the
                           covariates you regress onset/span against.
  3. weather_response() -- a quick per-species screen (OLS slope + Pearson r) of a
                           phenology metric against each covariate. Descriptive only
                           -- do proper mixed models (day, species, year) in R.

Open-Meteo notes:
  * Endpoint https://archive-api.open-meteo.com/v1/archive . No key. When `timezone`
    is passed, returned timestamps are naive LOCAL time (we localize them).
  * ERA5 archive lags ~5 days; for very recent mornings switch source="forecast"
    (uses the forecast endpoint, which also serves recent past days).
  * Free tier is NON-COMMERCIAL, CC BY 4.0 (attribution required). ~10k calls/day.
    For commercial use, self-host Open-Meteo or use a paid tier.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_VARS = ["temperature_2m", "cloud_cover", "precipitation",
                "wind_speed_10m", "relative_humidity_2m"]


def fetch_hourly(latitude, longitude, start_date, end_date, tz,
                 variables=None, cache_path=None, source="archive", timeout=60):
    """Return an hourly weather DataFrame indexed by tz-aware local time.

    start_date/end_date are date or 'YYYY-MM-DD'. If cache_path exists it is read
    instead of hitting the network (and a fresh fetch is written there).
    """
    variables = variables or DEFAULT_VARS

    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path)
        t = pd.to_datetime(df["time"])
        df["time"] = (t.dt.tz_localize(tz) if t.dt.tz is None else t.dt.tz_convert(tz))
        return df.set_index("time").sort_index()

    import requests  # imported lazily so the rest of the package needs no network
    url = FORECAST_URL if source == "forecast" else ARCHIVE_URL
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": str(start_date), "end_date": str(end_date),
        "hourly": ",".join(variables), "timezone": tz,
    }
    r = requests.get(url, params=params, timeout=timeout)
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    df = df.set_index("time").sort_index()

    if cache_path:
        out = df.copy()
        out.index = out.index.tz_localize(None)  # store naive local
        out.reset_index().to_csv(cache_path, index=False)
    return df


def _nearest(hourly: pd.DataFrame, when) -> pd.Series | None:
    if hourly.empty:
        return None
    idx = hourly.index.get_indexer([when], method="nearest")[0]
    return hourly.iloc[idx]


def morning_weather(hourly: pd.DataFrame, solar, dates, config=None) -> pd.DataFrame:
    """One weather row per morning: value at anchor + window aggregates.

    `solar` is a SolarModel; `dates` an iterable of datetime.date. Aggregates cover
    the same [window_start, window_end] the phenology uses, so covariates describe
    exactly the singing window.
    """
    from .phenology import DEFAULTS
    cfg = {**DEFAULTS, **(config or {})}
    anchor = cfg["anchor"]
    w0, w1 = cfg["window_start_min"], cfg["window_end_min"]

    rows = []
    for d in pd.Index(pd.unique(pd.to_datetime(list(dates)).date)):
        anchor_dt = solar.dawn(d) if anchor == "dawn" else solar.sunrise(d)
        lo, hi = anchor_dt + timedelta(minutes=w0), anchor_dt + timedelta(minutes=w1)
        sub = hourly[(hourly.index >= lo) & (hourly.index <= hi)]
        at = _nearest(hourly, anchor_dt)
        row = {"date": d}
        if at is not None:
            row["temp_at_anchor"] = float(at.get("temperature_2m", np.nan))
            row["cloud_at_anchor"] = float(at.get("cloud_cover", np.nan))
        if not sub.empty:
            if "temperature_2m" in sub:  row["temperature_2m"] = float(sub["temperature_2m"].mean())
            if "cloud_cover" in sub:     row["cloud_cover"] = float(sub["cloud_cover"].mean())
            if "precipitation" in sub:   row["precip_sum"] = float(sub["precipitation"].sum())
            if "wind_speed_10m" in sub:  row["wind_speed_10m"] = float(sub["wind_speed_10m"].mean())
            if "relative_humidity_2m" in sub: row["rh_mean"] = float(sub["relative_humidity_2m"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def attach_weather(morning_summary: pd.DataFrame, morning_wx: pd.DataFrame) -> pd.DataFrame:
    """Left-merge per-morning weather onto the per-species-morning summary."""
    ms = morning_summary.copy()
    ms["_d"] = pd.to_datetime(ms["date"]).dt.date
    wx = morning_wx.copy()
    wx["_d"] = pd.to_datetime(wx["date"]).dt.date
    merged = ms.merge(wx.drop(columns=["date"]), on="_d", how="left")
    return merged.drop(columns=["_d"])


def weather_response(summary_wx: pd.DataFrame, covariates=("temperature_2m", "cloud_cover"),
                     metric="onset_min", min_n=8) -> pd.DataFrame:
    """Per-species OLS slope + Pearson r of `metric` vs each covariate.

    Descriptive screen only (no p-values / no pooling). Negative onset~temperature
    slope = earlier singing on warmer mornings, the classic expectation.
    """
    rows = []
    for sp, g in summary_wx.groupby("common_name"):
        for cov in covariates:
            if cov not in g:
                continue
            d = g[[metric, cov]].dropna()
            if len(d) < min_n or d[cov].nunique() < 3:
                continue
            x, y = d[cov].to_numpy(float), d[metric].to_numpy(float)
            slope, intercept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            rows.append(dict(common_name=sp, covariate=cov, metric=metric, n=len(d),
                             slope=float(slope), pearson_r=r, r2=r * r))
    return (pd.DataFrame(rows)
            .sort_values(["common_name", "covariate"]).reset_index(drop=True))
