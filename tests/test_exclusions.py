"""Curated exclusions remove a known non-bird source without removing the species.

A recorder hears whatever is loud. Rain on an upturned bucket ran for four hours on
2026-08-17 and BirdNET read the plink as Wild Turkey, with woodpecker drumming as the
runner-up. The fix has to be surgical in two directions at once: drop those species on
that date, and leave the same species alone on every other morning, where they are real.

Nothing here is automatic. A 2000x spike can also be a real event -- a flock arriving,
a roost forming -- so a filter that quietly deleted outliers would be worse than the
noise it removed. Rules are declared by hand in deployments.json with a reason, and the
returned notes are what makes the removal visible in the build log and on the page.
"""
from __future__ import annotations

import json

import pandas as pd

import config


def _cfg(tmp_path, rules):
    p = tmp_path / "deployments.json"
    p.write_text(json.dumps({"sites": {"s": {
        "name": "S", "lat": 42.5, "lon": -72.5, "tz": "America/New_York", "primary": "a",
        "deployments": {"a": {"audio": "data", "recorder": "song-meter-micro-2"}},
        "exclusions": rules}}}), encoding="utf-8")
    return p


def _det():
    rows = [("2026-08-17", "Wild Turkey"), ("2026-08-17", "Wild Turkey"),
            ("2026-08-17", "Veery"), ("2026-08-16", "Wild Turkey"),
            ("2026-08-16", "Veery")]
    return pd.DataFrame({
        "datetime": pd.to_datetime([f"{d} 05:30:00" for d, _ in rows]),
        "common_name": [s for _, s in rows]})


RULE = [{"date": "2026-08-17", "species": ["Wild Turkey"], "reason": "bucket"}]


def test_drops_only_the_named_species_on_the_named_date(tmp_path):
    out, notes = config.apply_exclusions(_det(), "s", _cfg(tmp_path, RULE))
    assert len(out) == 3
    kept = set(zip(out["datetime"].dt.strftime("%Y-%m-%d"), out["common_name"]))
    assert ("2026-08-16", "Wild Turkey") in kept, "real turkeys on other days must survive"
    assert ("2026-08-17", "Veery") in kept, "other species that morning must survive"
    assert ("2026-08-17", "Wild Turkey") not in kept


def test_reports_what_it_removed(tmp_path):
    """The count and reason travel with the result; a silent drop reads as a bug."""
    _, notes = config.apply_exclusions(_det(), "s", _cfg(tmp_path, RULE))
    assert notes == [{"date": "2026-08-17", "species": ["Wild Turkey"],
                      "removed": 2, "reason": "bucket"}]


def test_a_rule_that_matches_nothing_produces_no_note(tmp_path):
    rule = [{"date": "2026-01-01", "species": ["Wild Turkey"], "reason": "n/a"}]
    out, notes = config.apply_exclusions(_det(), "s", _cfg(tmp_path, rule))
    assert len(out) == 5 and notes == []


def test_no_rules_is_a_pass_through(tmp_path):
    det = _det()
    out, notes = config.apply_exclusions(det, "s", _cfg(tmp_path, []))
    assert notes == [] and len(out) == len(det)


def test_unknown_site_is_not_an_error(tmp_path):
    """Deployments without exclusions must build exactly as before."""
    out, notes = config.apply_exclusions(_det(), "nope", _cfg(tmp_path, RULE))
    assert notes == [] and len(out) == 5


def test_the_shipped_config_is_readable(tmp_path):
    """Guards against a typo in the real deployments.json going unnoticed."""
    for r in config.exclusions("montague"):
        assert r["date"] and r["species"] and r.get("reason"), r
