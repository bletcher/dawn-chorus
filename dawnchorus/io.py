"""
Detection loader with automatic schema detection.

Different stations expose different schemas, and BirdNET-Go itself has migrated
column layouts over time. Rather than hardcode, we introspect the SQLite file,
find the table that looks like a detection log, and map its columns onto a
normalized frame:

    datetime, date, scientific_name, common_name, confidence, latitude, longitude

Known layouts handled:
  * BirdNET-Go  : table `notes` (or newer `detections`) with snake_case columns
                  (scientific_name, common_name, confidence, date, time, latitude, longitude)
  * BirdNET-Pi  : table `detections` with Sci_Name, Com_Name, Confidence, Date, Time, Lat, Lon
Anything else that carries the same *concepts* under different names will still work
as long as the aliases below cover it; extend ALIASES if your station differs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Candidate source-column names (lowercased) for each normalized field.
ALIASES = {
    "date": ["date"],
    "time": ["time"],
    "scientific_name": ["scientific_name", "sci_name", "scientificname"],
    "common_name": ["common_name", "com_name", "commonname"],
    "confidence": ["confidence", "conf"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
}

# Tables tried first (in order) when several look like detection logs.
PREFERRED_TABLES = ["detections", "notes", "birds"]

REQUIRED = ["date", "time", "scientific_name", "confidence"]


def _tables(con: sqlite3.Connection) -> list[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [r[1] for r in rows]


def _map_columns(cols: list[str]) -> dict | None:
    """Return {normalized_field: actual_column} if all REQUIRED fields resolve."""
    lower = {c.lower(): c for c in cols}
    mapping = {}
    for field, names in ALIASES.items():
        for n in names:
            if n in lower:
                mapping[field] = lower[n]
                break
    if all(f in mapping for f in REQUIRED):
        return mapping
    return None


def _pick_table(con: sqlite3.Connection) -> tuple[str, dict]:
    candidates = {}
    for t in _tables(con):
        m = _map_columns(_columns(con, t))
        if m:
            candidates[t] = m
    if not candidates:
        raise ValueError(
            "No table with date/time/scientific_name/confidence found. "
            f"Tables present: {_tables(con)}"
        )
    for pref in PREFERRED_TABLES:
        if pref in candidates:
            return pref, candidates[pref]
    # Otherwise take the largest table (most rows) as the detection log.
    best = max(candidates, key=lambda t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
    return best, candidates[best]


def load_detections(
    db_path: str | Path,
    min_confidence: float = 0.0,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """Load a BirdNET SQLite database into a normalized detections frame.

    latitude/longitude, if given, override whatever is in the DB (BirdNET-Go
    stores 0.0 when location filtering is disabled, which is unusable for solar
    calculations). If not given, per-row values from the DB are used.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        table, cmap = _pick_table(con)
        select = ", ".join(f'"{cmap[f]}" AS {f}' for f in cmap)
        df = pd.read_sql(f'SELECT {select} FROM "{table}"', con)
    finally:
        con.close()

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"] >= min_confidence].copy()

    # Combine date + time into a single naive local timestamp.
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime", "scientific_name"])
    df["date"] = df["datetime"].dt.date

    for col, val in (("latitude", latitude), ("longitude", longitude)):
        if val is not None:
            df[col] = val
        elif col not in df.columns:
            df[col] = pd.NA

    if "common_name" not in df.columns:
        df["common_name"] = df["scientific_name"]

    df["_source_table"] = table
    keep = ["datetime", "date", "scientific_name", "common_name",
            "confidence", "latitude", "longitude", "_source_table"]
    return df[keep].sort_values("datetime").reset_index(drop=True)
