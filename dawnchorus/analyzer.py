"""
Load BirdNET-Analyzer batch output into the normalized detections frame.

The live-station path (`io.load_detections`) reads a BirdNET-Pi/Go SQLite DB. The
BATCH path instead records only the dawn window (e.g. an AudioMoth or Song Meter on a
sunrise-relative schedule), then runs **BirdNET-Analyzer** over the folder of recordings
and gets one result table per file. Those tables carry each detection's OFFSET WITHIN ITS
FILE, not a wall-clock time, so we reconstruct the timestamp as:

    detection_time = <recording start, parsed from the filename> + <offset seconds>

and emit exactly the frame `io.load_detections` returns, so everything downstream (solar
anchoring, phenology, ECDF, weather) is byte-for-byte identical between the two paths.

Filenames: recorders stamp the start time into the name — AudioMoth `20260517_043000.WAV`,
Song Meter `PREFIX_20260517_043000.wav`. We pull a `YYYYMMDD_HHMMSS` substring by default;
override `ts_regex` / `ts_format` for other conventions.

TIMEZONE GOTCHA: **AudioMoth stamps filenames in UTC by default.** dawnchorus treats the
returned `datetime` as station-LOCAL. If your recorder's clock/filenames are in UTC (or any
tz other than the station's), pass `file_tz=` and `tz=` so we convert — otherwise solar
time is silently wrong by your whole UTC offset.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

# Default AudioMoth / Song Meter timestamp convention embedded in the filename.
TS_REGEX = r"\d{8}_\d{6}"
TS_FORMAT = "%Y%m%d_%H%M%S"


def _norm(col) -> str:
    """Collapse a header to comparable form: 'Begin Time (s)' -> 'begintimes'."""
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# Candidate columns (normalized), in preference order. BirdNET-Analyzer's exact
# headers vary by version and --rtype, so we introspect rather than hardcode.
_OFFSET_KEYS = ["fileoffsets", "starts", "starttimes", "begintimes", "start"]
_CONF_KEYS = ["confidence", "conf"]
_SCI_KEYS = ["scientificname", "sciname"]
_COM_KEYS = ["commonname", "comname"]
_FILE_KEYS = ["beginpath", "file", "infile", "sourcefile", "filepath", "path"]


def _pick(normmap: dict, keys: list[str]):
    for k in keys:
        if k in normmap:
            return normmap[k]
    return None


def _result_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return [p]
    files = sorted(q for q in p.rglob("*") if q.suffix.lower() in {".csv", ".txt", ".tsv"})
    if not files:
        raise FileNotFoundError(f"No .csv/.txt/.tsv result tables under {p}")
    return files


def _file_start(text, ts_regex: str, ts_format: str):
    m = re.search(ts_regex, str(text))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), ts_format)
    except ValueError:
        return None


def load_birdnet_analyzer(
    results_path: str | Path,
    min_confidence: float = 0.0,
    latitude: float | None = None,
    longitude: float | None = None,
    tz: str | None = None,
    file_tz: str | None = None,
    ts_regex: str = TS_REGEX,
    ts_format: str = TS_FORMAT,
) -> pd.DataFrame:
    """Read a directory (or single file) of BirdNET-Analyzer result tables.

    Returns the normalized detections frame (same columns as `load_detections`):
    datetime, date, scientific_name, common_name, confidence, latitude, longitude.

    `file_tz` (e.g. "UTC" for AudioMoth) plus `tz` (the station tz) convert filename
    timestamps to station-local time. Leave both None if filenames are already local.
    """
    frames = []
    for rf in _result_files(results_path):
        sep = "\t" if rf.suffix.lower() in {".txt", ".tsv"} else ","
        try:
            df = pd.read_csv(rf, sep=sep)
        except Exception:
            continue  # not a readable table (stray file in the results dir); skip
        if df.empty:
            continue

        normmap = {_norm(c): c for c in df.columns}
        off_c = _pick(normmap, _OFFSET_KEYS)
        conf_c = _pick(normmap, _CONF_KEYS)
        sci_c = _pick(normmap, _SCI_KEYS)
        com_c = _pick(normmap, _COM_KEYS)
        file_c = _pick(normmap, _FILE_KEYS)
        if off_c is None or conf_c is None or (sci_c is None and com_c is None):
            continue  # doesn't look like a BirdNET result table

        sub = pd.DataFrame()
        sub["_offset"] = pd.to_numeric(df[off_c], errors="coerce")
        sub["confidence"] = pd.to_numeric(df[conf_c], errors="coerce")
        sub["scientific_name"] = df[sci_c] if sci_c is not None else df[com_c]
        sub["common_name"] = df[com_c] if com_c is not None else df[sci_c]

        # Recording start: a per-row file column if the table has one (combined
        # output), else the result file's own name (per-file output).
        if file_c is not None:
            starts = df[file_c].map(lambda v: _file_start(Path(str(v)).name, ts_regex, ts_format))
        else:
            starts = pd.Series([_file_start(rf.name, ts_regex, ts_format)] * len(df), index=df.index)
        sub["_start"] = pd.to_datetime(starts.to_numpy(), errors="coerce")
        frames.append(sub)

    if not frames:
        raise ValueError(
            "No BirdNET-Analyzer detections found: no table under "
            f"{results_path} had offset + confidence + species columns. "
            "Re-run BirdNET-Analyzer with --rtype csv, or point at the right folder."
        )

    det = pd.concat(frames, ignore_index=True)
    bad_ts = int(det["_start"].isna().sum())
    det = det.dropna(subset=["_offset", "_start", "scientific_name"])
    if bad_ts:
        warnings.warn(
            f"{bad_ts} detections had no parseable {ts_format!r} timestamp in their "
            f"filename and were dropped; pass ts_regex/ts_format if your recorder names "
            f"files differently.", stacklevel=2,
        )
    det = det[det["confidence"] >= min_confidence].copy()

    dt = det["_start"] + pd.to_timedelta(det["_offset"], unit="s")
    if file_tz is not None:
        if tz is None:
            raise ValueError("file_tz given but tz (station tz) is None; both are needed to convert.")
        dt = (dt.dt.tz_localize(file_tz, ambiguous="NaT", nonexistent="shift_forward")
                .dt.tz_convert(tz).dt.tz_localize(None))
    det["datetime"] = dt
    det = det.dropna(subset=["datetime"])
    det["date"] = det["datetime"].dt.date

    det["latitude"] = latitude if latitude is not None else pd.NA
    det["longitude"] = longitude if longitude is not None else pd.NA
    det["_source_table"] = "birdnet-analyzer"

    keep = ["datetime", "date", "scientific_name", "common_name",
            "confidence", "latitude", "longitude", "_source_table"]
    return det[keep].sort_values("datetime").reset_index(drop=True)
