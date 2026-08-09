"""Recorder profiles: filename conventions, clock zones, and per-detection tagging.

The failures these guard against are silent ones — a mis-parsed filename drops a whole
recording, and a mis-assumed clock shifts every solar time by the UTC offset without
raising anything.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dawnchorus import load_birdnet_analyzer, recorders as rec


def _write(path, text):
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


ROWS = """
Start (s),End (s),Scientific name,Common name,Confidence
120.0,123.0,Turdus migratorius,American Robin,0.85
"""


# --- registry ------------------------------------------------------------------------

def test_known_profiles_expose_their_convention_and_clock():
    sm = rec.get("song-meter-micro-2")
    assert sm.file_tz is None                    # station-local clock -> no conversion
    assert sm.timestamp_spec() == (r"\d{8}_\d{6}", "%Y%m%d_%H%M%S")
    assert sm.nyquist == 12000                   # 24 kHz mono
    assert rec.get("audiomoth").file_tz == "UTC"


def test_unknown_recorder_id_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown recorder"):
        rec.get("no-such-box")


def test_get_none_means_caller_keeps_its_defaults():
    assert rec.get(None) is None


# --- sniffing ------------------------------------------------------------------------

@pytest.mark.parametrize("name,expect", [
    ("2MM43813_20260725_043500.wav", "compact"),          # Song Meter Micro 2, real name
    ("e74d3_2026-08-06_T05-43-03.wav", "iso-dash-t"),     # Owl Sense unit e74d3, real name
    ("2026-07-25_04-35-00.wav", "iso-dash"),
    ("20260725T043500.WAV", "compact-t"),
    ("5E0C1F80.WAV", "hex-epoch"),
])
def test_sniff_identifies_each_convention(name, expect):
    assert rec.sniff([name]).id == expect


def test_owl_sense_real_filenames_parse():
    """Regression: the `_T` separator matches no single-separator pattern, so before
    `iso-dash-t` existed every one of these files was silently dropped."""
    r = rec.get("owl-sense")
    regex, fmt = r.timestamp_spec()
    conv = rec.CONVENTIONS_BY_ID[r.convention]
    assert conv.parse("e74d3_2026-08-06_T05-43-03.wav") == pd.Timestamp("2026-08-06 05:43:03")
    assert conv.parse("e74d3_2026-08-07_T03-43-28.wav") == pd.Timestamp("2026-08-07 03:43:28")
    assert (regex, fmt) == (r"\d{4}-\d{2}-\d{2}_T\d{2}-\d{2}-\d{2}", "%Y-%m-%d_T%H-%M-%S")


def test_song_meter_names_do_not_match_the_owl_pattern():
    """The two profiles must not be able to claim each other's files."""
    owl = rec.CONVENTIONS_BY_ID["iso-dash-t"]
    assert owl.parse("2MM43813_20260806_044700.wav") is None


def test_sniff_returns_none_when_nothing_parses():
    assert rec.sniff(["no_timestamp_here.wav"]) is None


def test_profile_without_a_convention_resolves_by_sniffing():
    # `generic` ships with convention=None precisely so unknown hardware needs no code change.
    r = rec.get("generic").resolve(["BOX_2026-08-05_04-30-00.wav"])
    assert r.convention == "iso-dash"


def test_unresolvable_names_fall_back_to_the_historical_default():
    r = rec.get("generic").resolve(["mystery.wav"])
    assert r.convention == rec.DEFAULT_CONVENTION.id


# --- loader integration --------------------------------------------------------------

def test_recorder_column_tags_every_detection(tmp_path):
    _write(tmp_path / "2MM43813_20260725_043500.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, recorder="song-meter-micro-2")
    assert set(det["recorder"]) == {"song-meter-micro-2"}


def test_no_recorder_leaves_the_column_null_and_behaviour_unchanged(tmp_path):
    _write(tmp_path / "2MM43813_20260725_043500.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path)
    assert det["recorder"].isna().all()
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-07-25 04:37:00")


def test_profile_supplies_the_utc_clock_without_an_explicit_file_tz(tmp_path):
    # The whole point of a profile: you can't forget --file-tz for a UTC recorder.
    _write(tmp_path / "20260517_090000.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, tz="America/New_York", recorder="audiomoth")
    # 09:00 UTC + 120 s offset -> 05:02 EDT
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-17 05:02:00")


def test_explicit_file_tz_overrides_the_profile(tmp_path):
    _write(tmp_path / "20260517_090000.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, tz="America/New_York", file_tz=None,
                                recorder="song-meter-micro-2")
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-17 09:02:00")   # no conversion


def test_sniffed_convention_parses_an_unregistered_naming_scheme(tmp_path):
    # An ISO-style name would be dropped by the default regex; an unpinned profile
    # sniffs it instead, so the recording survives without a registry edit.
    _write(tmp_path / "BOX_2026-08-05_04-30-00.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, recorder="generic")
    assert len(det) == 1
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-08-05 04:32:00")


def test_owl_sense_results_load_end_to_end(tmp_path):
    """The real naming scheme survives the full loader, not just the regex."""
    _write(tmp_path / "e74d3_2026-08-06_T05-43-03.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, recorder="owl-sense")
    assert len(det) == 1                                    # 05:43:03 + the 120 s row offset
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-08-06 05:45:03")
    assert det["recorder"].iloc[0] == "owl-sense"


def test_hex_epoch_filenames_reconstruct_utc_times(tmp_path):
    # 0x5E0C89C0 = 1577880000 = 2020-01-01 12:00:00 UTC -> 07:00 EST, + the 120 s row offset
    _write(tmp_path / "5E0C89C0.results.csv", ROWS)
    det = load_birdnet_analyzer(tmp_path, tz="America/New_York", recorder="audiomoth-legacy")
    assert det["datetime"].iloc[0] == pd.Timestamp("2020-01-01 07:02:00")


def test_two_recorders_stay_separable_after_concatenation(tmp_path):
    """The reason the column exists: one site, two boxes, still distinguishable."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a / "2MM43813_20260805_043000.results.csv", ROWS)
    _write(b / "e74d3_2026-08-05_T04-30-00.results.csv", ROWS)
    both = pd.concat([load_birdnet_analyzer(a, recorder="song-meter-micro-2"),
                      load_birdnet_analyzer(b, recorder="owl-sense")])
    assert both.groupby("recorder").size().to_dict() == {"owl-sense": 1, "song-meter-micro-2": 1}
