"""Loader: schema auto-detection across station layouts, filtering, overrides."""
from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from dawnchorus import load_detections


def test_detects_birdnet_go_notes_table(birdnet_go_db):
    df = load_detections(birdnet_go_db)  # min_confidence=0.0 keeps all
    assert list(df.columns[:5]) == ["datetime", "date", "scientific_name",
                                    "common_name", "confidence"]
    assert df["_source_table"].unique().tolist() == ["notes"]
    assert set(df["scientific_name"]) == {"turdus_migratorius", "melospiza_melodia"}
    # date + time were combined into a single local timestamp, sorted ascending.
    assert df["datetime"].is_monotonic_increasing
    assert df["date"].iloc[0] == date(2025, 5, 1)


def test_detects_birdnet_pi_detections_table(birdnet_pi_db):
    df = load_detections(birdnet_pi_db)
    assert df["_source_table"].unique().tolist() == ["detections"]
    assert "American Robin" in set(df["common_name"])
    assert df["confidence"].dtype.kind == "f"  # coerced to numeric


def test_confidence_filter_drops_low_rows(birdnet_go_db):
    df = load_detections(birdnet_go_db, min_confidence=0.6)
    # The 0.40 robin detection is dropped; 0.90 robin and 0.75 sparrow remain.
    assert len(df) == 2
    assert (df["confidence"] >= 0.6).all()


def test_latlon_override_replaces_stored_zeros(birdnet_go_db):
    df = load_detections(birdnet_go_db, latitude=42.53, longitude=-72.53)
    assert (df["latitude"] == 42.53).all()
    assert (df["longitude"] == -72.53).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detections(tmp_path / "nope.db")


def test_unrecognised_schema_raises(tmp_path):
    db = tmp_path / "weird.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE misc (foo TEXT, bar TEXT)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="No table"):
        load_detections(db)
