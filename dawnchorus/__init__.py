"""dawnchorus: dawn-chorus phenology from BirdNET detection databases."""

from .io import load_detections
from .analyzer import load_birdnet_analyzer
from .solar import SolarModel
from .phenology import morning_summary, species_phenology, DEFAULTS
from .seasonal import composition, richness
from .weather import (fetch_hourly, morning_weather, attach_weather,
                      weather_response, DEFAULT_VARS)
from .ecdf import species_ecdf, ecdf_quantiles, ecdf_distance
from .recorders import Recorder, REGISTRY as RECORDERS

__all__ = [
    "load_detections", "load_birdnet_analyzer", "SolarModel", "morning_summary",
    "species_phenology", "composition", "richness", "DEFAULTS", "fetch_hourly",
    "morning_weather", "attach_weather", "weather_response", "species_ecdf",
    "ecdf_quantiles", "ecdf_distance", "run", "Recorder", "RECORDERS",
    "CHART_MIN_CONFIDENCE",
]
__version__ = "0.5.0"

#: Confidence floor for everything CHARTED or PUBLISHED.
#:
#: 0.40, not BirdNET's conventional 0.50. On this archive the median detection confidence
#: is 0.465 -- the bulk of real vocalisations sit just under the 0.50 line, 55% of what the
#: analyser captured never reached a chart, and 22 plausible species existed only between
#: 0.25 and 0.50. The cost is a longer false-positive tail, which is what the curated
#: exclusions in deployments.json are for.
#:
#: Inference captures at 0.25 (track.py --capture-conf), so this can move again without
#: re-running the models. It changes every n and every onset, so move it for the WHOLE
#: archive at once or the series stops being comparable.
CHART_MIN_CONFIDENCE = 0.40


def run(db_path=None, latitude=None, longitude=None, tz=None,
        min_confidence=CHART_MIN_CONFIDENCE,
        config=None, weather=False, weather_cache=None, weather_source="archive",
        analyzer_path=None, file_tz=None, recorder=None):
    """End-to-end: a detection source -> dict of tidy result frames.

    Pass exactly one source: `db_path` (BirdNET-Pi/Go SQLite, the live-station path) or
    `analyzer_path` (a folder/file of BirdNET-Analyzer result tables, the batch path).
    For the batch path, `file_tz` (e.g. "UTC" for AudioMoth) converts filename timestamps
    to the station's `tz`.

    `recorder` names a hardware profile (`dawnchorus.recorders`); it supplies the filename
    convention and clock zone, and tags every detection so one site can carry more than
    one box without pooling their biases.

    With weather=True, per-morning Open-Meteo covariates are fetched (or read from
    weather_cache) and merged onto morning_summary, and a per-species onset~weather
    screen is produced.
    """
    if bool(db_path) == bool(analyzer_path):
        raise ValueError("pass exactly one of db_path or analyzer_path")
    if db_path:
        det = load_detections(db_path, min_confidence=min_confidence,
                              latitude=latitude, longitude=longitude, recorder=recorder)
    else:
        det = load_birdnet_analyzer(analyzer_path, min_confidence=min_confidence,
                                    latitude=latitude, longitude=longitude,
                                    tz=tz, file_tz=file_tz, recorder=recorder)
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
