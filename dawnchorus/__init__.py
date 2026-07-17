"""dawnchorus: dawn-chorus phenology from BirdNET detection databases."""

from .io import load_detections
from .analyzer import load_birdnet_analyzer
from .solar import SolarModel
from .phenology import morning_summary, species_phenology, DEFAULTS
from .seasonal import composition, richness
from .weather import (fetch_hourly, morning_weather, attach_weather,
                      weather_response, DEFAULT_VARS)
from .ecdf import species_ecdf, ecdf_quantiles, ecdf_distance

__all__ = [
    "load_detections", "load_birdnet_analyzer", "SolarModel", "morning_summary",
    "species_phenology", "composition", "richness", "DEFAULTS", "fetch_hourly",
    "morning_weather", "attach_weather", "weather_response", "species_ecdf",
    "ecdf_quantiles", "ecdf_distance", "run",
]
__version__ = "0.4.0"


def run(db_path=None, latitude=None, longitude=None, tz=None, min_confidence=0.5,
        config=None, weather=False, weather_cache=None, weather_source="archive",
        analyzer_path=None, file_tz=None):
    """End-to-end: a detection source -> dict of tidy result frames.

    Pass exactly one source: `db_path` (BirdNET-Pi/Go SQLite, the live-station path) or
    `analyzer_path` (a folder/file of BirdNET-Analyzer result tables, the batch path).
    For the batch path, `file_tz` (e.g. "UTC" for AudioMoth) converts filename timestamps
    to the station's `tz`.

    With weather=True, per-morning Open-Meteo covariates are fetched (or read from
    weather_cache) and merged onto morning_summary, and a per-species onset~weather
    screen is produced.
    """
    if bool(db_path) == bool(analyzer_path):
        raise ValueError("pass exactly one of db_path or analyzer_path")
    if db_path:
        det = load_detections(db_path, min_confidence=min_confidence,
                              latitude=latitude, longitude=longitude)
    else:
        det = load_birdnet_analyzer(analyzer_path, min_confidence=min_confidence,
                                    latitude=latitude, longitude=longitude,
                                    tz=tz, file_tz=file_tz)
    solar = SolarModel(latitude, longitude, tz)
    det = solar.annotate(det)

    ms = morning_summary(det, config)
    out = {
        "detections": det,
        "morning_summary": ms,
        "species_phenology_month": species_phenology(ms, by="month"),
        "composition_month": composition(det, by="month", config=config),
        "richness_month": richness(det, by="month", config=config),
    }

    ecdf = species_ecdf(det, by="month", config=config)
    out["species_ecdf_month"] = ecdf
    out["ecdf_quantiles_month"] = ecdf_quantiles(ecdf, by="month")

    if weather:
        dates = sorted(det["date"].unique())
        hourly = fetch_hourly(latitude, longitude, min(dates), max(dates), tz,
                              cache_path=weather_cache, source=weather_source)
        wx = morning_weather(hourly, solar, dates, config=config)
        ms_wx = attach_weather(ms, wx)
        out["morning_summary"] = ms_wx
        out["morning_weather"] = wx
        out["weather_response"] = weather_response(ms_wx)

    return out
