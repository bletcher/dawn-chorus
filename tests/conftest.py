"""Shared pytest fixtures and path setup.

Makes the standalone synthetic-data generator in ``tools/`` importable so the
integration test can build a realistic station DB with a known planted signal.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# tools/ is a script directory, not a package -- put it on the path directly.
sys.path.insert(0, str(ROOT / "tools"))


def _make_db(path, table, colnames, rows):
    """Create a one-table SQLite DB. `colnames` maps our concept -> actual column.

    `rows` is a list of dicts keyed by concept (date, time, scientific_name,
    common_name, confidence, latitude, longitude).
    """
    con = sqlite3.connect(str(path))
    cols = ", ".join(f'"{c}" TEXT' for c in colnames.values())
    con.execute(f'CREATE TABLE "{table}" ({cols})')
    placeholders = ", ".join("?" for _ in colnames)
    keys = list(colnames.keys())
    con.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        [tuple(r.get(k) for k in keys) for r in rows],
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def birdnet_go_db(tmp_path):
    """A minimal BirdNET-Go layout: table `notes`, snake_case columns."""
    colnames = {
        "date": "date", "time": "time",
        "scientific_name": "scientific_name", "common_name": "common_name",
        "confidence": "confidence", "latitude": "latitude", "longitude": "longitude",
    }
    rows = [
        dict(date="2025-05-01", time="04:30:00", scientific_name="turdus_migratorius",
             common_name="American Robin", confidence="0.90", latitude="0.0", longitude="0.0"),
        dict(date="2025-05-01", time="05:15:00", scientific_name="turdus_migratorius",
             common_name="American Robin", confidence="0.40", latitude="0.0", longitude="0.0"),
        dict(date="2025-05-01", time="05:45:00", scientific_name="melospiza_melodia",
             common_name="Song Sparrow", confidence="0.75", latitude="0.0", longitude="0.0"),
    ]
    return _make_db(tmp_path / "go.db", "notes", colnames, rows)


@pytest.fixture
def birdnet_pi_db(tmp_path):
    """A minimal BirdNET-Pi layout: table `detections`, CamelCase columns."""
    colnames = {
        "date": "Date", "time": "Time",
        "scientific_name": "Sci_Name", "common_name": "Com_Name",
        "confidence": "Confidence", "latitude": "Lat", "longitude": "Lon",
    }
    rows = [
        dict(date="2025-05-01", time="04:30:00", scientific_name="Turdus migratorius",
             common_name="American Robin", confidence="0.88", latitude="42.5", longitude="-72.5"),
        dict(date="2025-05-01", time="05:45:00", scientific_name="Melospiza melodia",
             common_name="Song Sparrow", confidence="0.72", latitude="42.5", longitude="-72.5"),
    ]
    return _make_db(tmp_path / "pi.db", "detections", colnames, rows)
