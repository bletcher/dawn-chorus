"""The ladder's structure and the audio rung's three states.

Phase 1 of the Brush-and-Rungs restructure moved every chart into a named rung. The claim
was that nothing changed except the seasonal chart's home, and a large markup move is
exactly where that claim breaks quietly -- one dropped element and a control stops working
with no error anywhere.

The audio rung matters most because it differs BETWEEN BUILDS: recordings never leave the
machine, so the published payload carries no `audio` and rung 3 must not exist there at
all. That difference is invisible in any single page's markup, so it is checked by running
the page's real updateAudioCard against a stub in each state.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import build_site
import check_js

pytestmark = pytest.mark.skipif(not check_js.have_node(),
                                reason="node is not installed; cannot run the emitted JS")
SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "check_ui.js"


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """Twelve mornings spanning three ISO weeks and a month boundary.

    The brush has to be exercised over a record wide enough to hold ranges, and across the
    boundaries where snapping is easiest to get wrong -- a week that straddles the end of a
    month is exactly where an off-by-one in periodBounds would hide.
    """
    src = tmp_path_factory.mktemp("results")
    offs = [37, 61, 62, 149, 210, 211, 640, 1811, 3007, 5400]
    for day in ("20260525", "20260526", "20260527", "20260529", "20260530", "20260531",
                "20260601", "20260602", "20260604", "20260605", "20260608", "20260609"):
        lines = ["Start (s),End (s),Scientific name,Common name,Confidence"]
        for i, o in enumerate(offs):
            lines.append(f"{o}.0,{o+3}.0,Turdus migratorius,American Robin,"
                         f"{0.55 + (i % 4) * 0.08:.2f}")
        (src / f"{day}_043000.BirdNET.results.csv").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")
    out = tmp_path_factory.mktemp("ladder") / "dash.html"
    build_site.main(["--from-analyzer", str(src), "--out", str(out),
                     "--lat", "42.5372", "--lon", "-72.5317", "--tz", "America/New_York"])
    return out


def test_a_single_morning_page_still_passes(tmp_path):
    """A station on its first morning is a valid page, not a broken one."""
    src = tmp_path / "one"
    src.mkdir()
    lines = ["Start (s),End (s),Scientific name,Common name,Confidence"]
    for o in (37, 61, 149, 210, 640):
        lines.append(f"{o}.0,{o+3}.0,Turdus migratorius,American Robin,0.71")
    (src / "20260517_043000.BirdNET.results.csv").write_text("\n".join(lines) + "\n",
                                                            encoding="utf-8")
    out = tmp_path / "one.html"
    build_site.main(["--from-analyzer", str(src), "--out", str(out),
                     "--lat", "42.5372", "--lon", "-72.5317", "--tz", "America/New_York"])
    r = subprocess.run(["node", str(SCRIPT), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_ladder_and_audio_states(page):
    r = subprocess.run(["node", str(SCRIPT), str(page)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout, r.stdout


def test_every_card_and_finding_survived(page):
    """Nine cards and seven computed findings, same as before the restructure."""
    src = page.read_text(encoding="utf-8")
    cards = set(re.findall(r'data-card="([\w-]+)"', src))
    assert cards == {"trend", "period", "ecdf", "heat",
                     "season", "temp", "rain", "table"}, sorted(cards)
    assert 'id="chart-hero"' in src, "the score should be the hero"
    assert 'data-card="timeline"' not in src, "the score was moved, not copied"
    finds = set(re.findall(r'id="find-(\w+)"', src))
    assert finds == {"overview", "period", "morning", "shape",
                     "season", "weather", "species"}, sorted(finds)


def test_the_checker_fails_on_a_dead_locked_rung(page, tmp_path):
    """A guard that cannot fail is not a guard.

    Reintroduce the pre-restructure behaviour, where only the CARD was hidden and the rung
    stayed on screen -- which on the published site would leave a rung nobody can ever open.
    """
    src = page.read_text(encoding="utf-8")
    broken = src.replace("if(!hasAudio()){ rung.hidden=true; return; }",
                         "if(!hasAudio()){ card.hidden=true; return; }", 1)
    assert broken != src, "patch point not found"
    p = tmp_path / "broken.html"
    p.write_text(broken, encoding="utf-8")
    r = subprocess.run(["node", str(SCRIPT), str(p)], capture_output=True, text=True)
    assert r.returncode != 0 and "FAIL" in r.stdout, r.stdout
