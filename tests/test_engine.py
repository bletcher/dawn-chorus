"""Engine layer + deployment config.

The engine's contract is that BOTH back ends emit an artefact indistinguishable from
BirdNET-Analyzer's own output, because the manifest, the loaders, the payload builder and
both dashboards all read it. These tests pin the artefact and the pool sizing; they do not
need the .tflite models, so they run in CI.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import config
import engine
from dawnchorus import load_birdnet_analyzer

DETS = [
    {"offset_s": 120.0, "scientific_name": "Turdus migratorius",
     "common_name": "American Robin", "confidence": 0.8512},
    {"offset_s": 3600.0, "scientific_name": "Melospiza melodia",
     "common_name": "Song Sparrow", "confidence": 0.4013},
]


# --- the artefact --------------------------------------------------------------------

def test_written_results_match_the_analyzer_layout(tmp_path):
    wav = tmp_path / "2MM43813_20260725_043500.wav"
    out = tmp_path / "r.csv"
    engine._write_results(out, DETS, wav)
    df = pd.read_csv(out)
    assert list(df.columns) == engine.HEADER
    assert df["Start (s)"].tolist() == [120.0, 3600.0]
    assert df["End (s)"].tolist() == [123.0, 3603.0]          # 3 s BirdNET window
    assert df["Confidence"].tolist() == [0.8512, 0.4013]


def test_file_column_holds_the_source_wav_not_the_result_name(tmp_path):
    """load_birdnet_analyzer PREFERS the File column over the result filename, so a wrong
    value here silently relocates every detection in time."""
    wav = tmp_path / "2MM43813_20260725_043500.wav"
    out = tmp_path / "totally_unrelated_name.csv"
    engine._write_results(out, DETS, wav)
    assert pd.read_csv(out)["File"].iloc[0] == str(wav.resolve())


def test_engine_output_round_trips_through_the_loader(tmp_path):
    """The whole point: what an engine writes, the pipeline reads back correctly."""
    wav = tmp_path / "2MM43813_20260725_043500.wav"
    engine._write_results(tmp_path / "x.BirdNET.results.csv", DETS, wav)
    det = load_birdnet_analyzer(tmp_path, recorder="song-meter-micro-2")
    assert len(det) == 2
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-07-25 04:37:00")   # 04:35 + 120 s
    assert det["datetime"].iloc[1] == pd.Timestamp("2026-07-25 05:35:00")   # 04:35 + 3600 s
    assert set(det["recorder"]) == {"song-meter-micro-2"}


def test_owl_sense_names_round_trip_too(tmp_path):
    wav = tmp_path / "e74d3_2026-08-06_T05-43-03.wav"
    engine._write_results(tmp_path / "y.BirdNET.results.csv", DETS[:1], wav)
    det = load_birdnet_analyzer(tmp_path, recorder="owl-sense")
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-08-06 05:45:03")


# --- pool sizing ---------------------------------------------------------------------

def test_explicit_jobs_is_respected():
    assert engine._plan_jobs([], 3) == 3


def test_auto_jobs_is_at_least_one_and_capped_by_cores(tmp_path):
    paths = [tmp_path / f"f{i}.wav" for i in range(64)]
    n = engine._plan_jobs(paths, 0)
    assert 1 <= n <= max(1, (os.cpu_count() or 4) // 2)


def test_memory_estimate_falls_back_when_files_are_unreadable(tmp_path):
    """Unreadable/absent files fall back to a sane default rather than estimating zero
    (which would let the planner start unlimited workers)."""
    assert engine._peak_gb_per_worker([tmp_path / "missing.wav"]) == engine.DEFAULT_PEAK_GB


def test_planner_matches_the_calibration_observations(monkeypatch, tmp_path):
    """Calibrated against real runs: 2 workers were fine on 1-hour files with ~10.7 GB
    free, 4 workers raised MemoryError at ~10.6 GB. The planner must reproduce that."""
    monkeypatch.setattr(engine, "_peak_gb_per_worker", lambda paths: 2.8)   # 1-hour file
    monkeypatch.setattr(engine, "_free_gb", lambda: 10.7)
    monkeypatch.setattr(engine.os, "cpu_count", lambda: 8)
    assert engine._plan_jobs([tmp_path / f"f{i}.wav" for i in range(4)], 0) == 2


def test_planner_shrinks_for_longer_recordings(monkeypatch, tmp_path):
    """A 2-hour Owl file costs ~2x a 1-hour Song Meter file, so the pool must shrink."""
    monkeypatch.setattr(engine, "_free_gb", lambda: 10.7)
    monkeypatch.setattr(engine.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(engine, "_peak_gb_per_worker", lambda paths: 5.6)
    assert engine._plan_jobs([tmp_path / f"f{i}.wav" for i in range(4)], 0) == 1


# --- engine selection ----------------------------------------------------------------

def test_unavailable_engine_explains_how_to_fix_it(monkeypatch):
    monkeypatch.setattr(engine, "have", lambda name: False)
    with pytest.raises(RuntimeError, match="not available"):
        engine.resolve("analyzer")
    with pytest.raises(RuntimeError, match="no inference engine available"):
        engine.resolve("auto")


# --- deployment config ---------------------------------------------------------------

def test_config_lists_both_recorders_at_montague():
    ds = config.deployments("montague")
    assert {d.key for d in ds} == {"sm", "owl"}
    by = {d.key: d for d in ds}
    assert by["sm"].recorder == "song-meter-micro-2"
    assert by["owl"].recorder == "owl-sense"
    assert by["owl"].audio.name == "data_owl"
    assert by["sm"].results.name == "results"


def test_primary_is_the_reference_recorder():
    assert config.primary("montague").key == "sm"


def test_config_rejects_an_unknown_recorder(tmp_path):
    """A typo'd recorder id must fail here, not become a silent mis-parse downstream."""
    bad = tmp_path / "deployments.json"
    bad.write_text('{"sites": {"s": {"name": "S", "lat": 1, "lon": 2, "tz": "UTC",'
                   ' "deployments": {"a": {"audio": "data", "recorder": "nope"}}}}}',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="unknown recorder"):
        config.deployments(path=bad)
