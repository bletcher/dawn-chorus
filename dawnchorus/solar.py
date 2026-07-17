"""
Solar-time anchoring.

The whole point of dawn-chorus analysis is that birds key their singing to the
*sun*, not to the clock. Two mornings a month apart have very different clock
sunrises, so comparing "detections at 05:30" across dates is meaningless. This
module converts every detection's local clock time into minutes relative to two
solar anchors:

    min_from_sunrise   : detection_time - sunrise      (negative = before sunrise)
    min_from_dawn      : detection_time - civil dawn    (civil twilight begin, sun 6 deg below horizon)

Civil dawn is usually the more biologically relevant anchor for onset, since many
songbirds begin 30-90 min before sunrise; sunrise is the more familiar anchor for
reporting. We compute both and let downstream code choose.
"""

from __future__ import annotations

import warnings
from datetime import date as _date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
from astral import LocationInfo
from astral.sun import sun


class SolarModel:
    """Compute sunrise / civil dawn for a fixed station location & timezone."""

    def __init__(self, latitude: float, longitude: float, tz: str):
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise ValueError(f"Implausible lat/lon: {latitude}, {longitude}")
        self.latitude = latitude
        self.longitude = longitude
        self.tz = ZoneInfo(tz)
        self._loc = LocationInfo(latitude=latitude, longitude=longitude)

    @lru_cache(maxsize=4096)
    def _events(self, d: _date) -> dict:
        # astral returns tz-aware datetimes in the tz we pass in. At high
        # latitudes the sun may never rise/set or never reach the 6-degree
        # civil-dawn depression (polar day/night); astral raises ValueError for
        # those events. We swallow it and return an empty dict so callers get
        # None for the missing anchor rather than a crash mid-season.
        try:
            return sun(self._loc.observer, date=d, tzinfo=self.tz)
        except ValueError:
            return {}

    def sunrise(self, d: _date) -> datetime | None:
        return self._events(d).get("sunrise")

    def dawn(self, d: _date) -> datetime | None:
        return self._events(d).get("dawn")  # civil dawn (6 deg depression)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add min_from_sunrise / min_from_dawn columns to a detections frame.

        Requires a tz-naive or tz-aware `datetime` column in station-local time.
        """
        out = df.copy()
        dt = pd.to_datetime(out["datetime"])
        # Localize naive timestamps to the station tz so subtraction is valid.
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(self.tz)
        else:
            dt = dt.dt.tz_convert(self.tz)
        out["datetime"] = dt

        # Compute solar events once per unique calendar date, then map back.
        uniq = pd.Index(dt.dt.date.unique())
        sr = {d: self.sunrise(d) for d in uniq}
        dw = {d: self.dawn(d) for d in uniq}
        det_date = dt.dt.date

        # Polar day/night dates yield None anchors -> NaN min_from_* -> those
        # detections drop out of every windowed analysis. Warn rather than fail.
        polar = [d for d in uniq if sr[d] is None or dw[d] is None]
        if polar:
            warnings.warn(
                f"{len(polar)} of {len(uniq)} dates have no civil dawn and/or "
                f"sunrise at lat {self.latitude:.2f} (polar day/night); their "
                f"detections get NaN solar time and are excluded from windowed "
                f"metrics. First affected: {min(polar)}.",
                stacklevel=2,
            )

        sr_series = pd.to_datetime(pd.Series(det_date.map(sr).values, index=out.index), utc=True)
        dw_series = pd.to_datetime(pd.Series(det_date.map(dw).values, index=out.index), utc=True)
        dt_utc = dt.dt.tz_convert("UTC")

        out["sunrise"] = sr_series.dt.tz_convert(self.tz)
        out["dawn"] = dw_series.dt.tz_convert(self.tz)
        out["min_from_sunrise"] = (dt_utc.values - sr_series.values) / pd.Timedelta(minutes=1)
        out["min_from_dawn"] = (dt_utc.values - dw_series.values) / pd.Timedelta(minutes=1)
        return out
