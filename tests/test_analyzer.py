"""BirdNET-Analyzer batch adapter: timestamp reconstruction, formats, tz, e2e."""
from __future__ import annotations

import pandas as pd
import pytest

from dawnchorus import SolarModel, load_birdnet_analyzer, morning_summary


def _write(path, text):
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_per_file_csv_reconstructs_absolute_times(tmp_path):
    # Result filename carries the recording start; rows carry within-file offsets.
    _write(tmp_path / "20260517_043000.BirdNET.results.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
120.0,123.0,Turdus migratorius,American Robin,0.85
3600.0,3603.0,Melospiza melodia,Song Sparrow,0.40
""")
    det = load_birdnet_analyzer(tmp_path)
    assert len(det) == 2
    # 04:30:00 + 120 s = 04:32:00 ; + 3600 s = 05:30:00
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-17 04:32:00")
    assert det["datetime"].iloc[1] == pd.Timestamp("2026-05-17 05:30:00")
    assert det["scientific_name"].iloc[0] == "Turdus migratorius"


def test_confidence_filter(tmp_path):
    _write(tmp_path / "20260517_043000.results.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
120.0,123.0,Turdus migratorius,American Robin,0.85
3600.0,3603.0,Melospiza melodia,Song Sparrow,0.40
""")
    det = load_birdnet_analyzer(tmp_path, min_confidence=0.6)
    assert len(det) == 1
    assert det["common_name"].iloc[0] == "American Robin"


def test_combined_csv_uses_per_row_file_column(tmp_path):
    _write(tmp_path / "combined.csv", """
Start (s),Scientific name,Common name,Confidence,File
60.0,Turdus migratorius,American Robin,0.9,/data/20260518_050000.WAV
30.0,Cardinalis cardinalis,Northern Cardinal,0.8,/data/20260519_051500.WAV
""")
    det = load_birdnet_analyzer(tmp_path)
    got = set(det["datetime"])
    assert pd.Timestamp("2026-05-18 05:01:00") in got   # 05:00:00 + 60 s
    assert pd.Timestamp("2026-05-19 05:15:30") in got   # 05:15:00 + 30 s


def test_raven_tab_table_prefers_file_offset_and_falls_back_to_common_name(tmp_path):
    # Tab-separated; File Offset (s) must win over Begin Time (s); Begin Path must win
    # over the result filename; no Scientific name -> fall back to Common Name.
    header = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
              "Species Code", "Common Name", "Confidence", "Begin Path", "File Offset (s)"]
    row = ["1", "Spectrogram 1", "1", "9999.0", "9999.0", "amerob", "American Robin",
           "0.7", "/x/20260521_060000.WAV", "120.0"]
    _write(tmp_path / "20260520_043000.BirdNET.selection.table.txt",
           "\t".join(header) + "\n" + "\t".join(row))
    det = load_birdnet_analyzer(tmp_path)
    assert len(det) == 1
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-21 06:02:00")  # 06:00 + 120 s
    assert det["scientific_name"].iloc[0] == "American Robin"
    assert det["common_name"].iloc[0] == "American Robin"


def test_timezone_conversion_from_utc_filenames(tmp_path):
    # AudioMoth default: filenames in UTC. 09:00 UTC in May -> 05:00 EDT (UTC-4).
    _write(tmp_path / "20260517_090000.results.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
0.0,3.0,Turdus migratorius,American Robin,0.9
""")
    det = load_birdnet_analyzer(tmp_path, tz="America/New_York", file_tz="UTC")
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-17 05:00:00")


def test_song_meter_filename_prefix_is_handled(tmp_path):
    # Wildlife Acoustics Song Meters name files PREFIX_YYYYMMDD_HHMMSS.wav; the result
    # table inherits that name. The default regex must find the timestamp after the prefix.
    _write(tmp_path / "SMU12345_20260522_050000.BirdNET.results.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
90.0,93.0,Hylocichla mustelina,Wood Thrush,0.80
""")
    det = load_birdnet_analyzer(tmp_path)
    assert det["datetime"].iloc[0] == pd.Timestamp("2026-05-22 05:01:30")  # 05:00:00 + 90 s


def test_unparseable_filename_warns_and_drops(tmp_path):
    _write(tmp_path / "no_timestamp_here.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
10.0,13.0,Turdus migratorius,American Robin,0.9
""")
    with pytest.warns(UserWarning, match="no parseable"):
        det = load_birdnet_analyzer(tmp_path)
    assert det.empty


def test_adapter_output_flows_through_solar_and_phenology(tmp_path):
    _write(tmp_path / "20260517_043000.results.csv", """
Start (s),End (s),Scientific name,Common name,Confidence
120.0,123.0,Turdus migratorius,American Robin,0.85
3600.0,3603.0,Melospiza melodia,Song Sparrow,0.55
""")
    det = load_birdnet_analyzer(tmp_path, latitude=42.53, longitude=-72.53)
    solar = SolarModel(42.53, -72.53, "America/New_York")
    det = solar.annotate(det)
    assert "min_from_dawn" in det.columns
    ms = morning_summary(det)
    assert set(ms["scientific_name"]) == {"Turdus migratorius", "Melospiza melodia"}
