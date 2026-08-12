"""Deployment audio folders may live off the repo (an external drive).

Card dumps are several GB per session, so `audio` has to accept an absolute path while
relative paths keep meaning "inside the repo". Getting this wrong is quiet: a deployment
pointed at the wrong folder simply looks empty, and a run then re-analyses everything.
"""
from __future__ import annotations

import json
import sys

import pytest

import config


def _cfg(tmp_path, audio):
    p = tmp_path / "deployments.json"
    p.write_text(json.dumps({"sites": {"s": {
        "name": "S", "lat": 42.5, "lon": -72.5, "tz": "America/New_York",
        "primary": "a",
        "deployments": {"a": {"audio": audio, "recorder": "song-meter-micro-2"}}}}}),
        encoding="utf-8")
    return p


def test_relative_audio_stays_inside_the_repo(tmp_path):
    d = config.deployments(path=_cfg(tmp_path, "data"))[0]
    assert d.audio == (config.ROOT / "data").resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="drive-letter path")
def test_absolute_windows_path_is_used_as_given(tmp_path):
    d = config.deployments(path=_cfg(tmp_path, "F:/dev/dawn-chorus/data"))[0]
    assert str(d.audio).replace("\\", "/").lower() == "f:/dev/dawn-chorus/data"
    assert not str(d.audio).lower().startswith(str(config.ROOT).lower())


@pytest.mark.skipif(sys.platform == "win32", reason="posix path")
def test_absolute_posix_path_is_used_as_given(tmp_path):
    d = config.deployments(path=_cfg(tmp_path, "/mnt/big/dawn-chorus/data"))[0]
    assert str(d.audio) == "/mnt/big/dawn-chorus/data"


def test_results_and_manifest_follow_the_audio_folder(tmp_path):
    """They live inside it, so a deployment moves as one unit."""
    d = config.deployments(path=_cfg(tmp_path, "F:/dev/dawn-chorus/data"
                                     if sys.platform == "win32" else "/mnt/big/data"))[0]
    assert d.results == d.audio / "results"
    assert d.manifest == d.audio / "manifest.json"
