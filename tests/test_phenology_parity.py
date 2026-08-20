"""The page recomputes phenology in JavaScript; it must agree with the Python it replaced.

The detection floor used to be baked in at build time. Making it adjustable moved
onset/offset/peak/occupancy into the browser, which means there are now TWO implementations
of `dawnchorus.phenology.morning_summary` -- and the second one is the only one anybody
looks at. If they drift, the dashboard silently stops matching the CSVs and the published
payload, and no chart would look wrong enough to notice.

tools/check_phen.js pulls the real functions out of the emitted page (not a copy of them)
and runs them against that page's own payload at the build-time floor, where Python's answer
is already in the file. Skipped without node, like the JS syntax check.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

import build_site
import check_js

pytestmark = pytest.mark.skipif(not check_js.have_node(),
                                reason="node is not installed; cannot run the emitted JS")
SCRIPT = __import__("pathlib").Path(__file__).resolve().parent.parent / "tools" / "check_phen.js"


def _run(*args):
    return subprocess.run(["node", str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """Two mornings deep enough to straddle the floor: one species well over it, one under.

    Times are spread unevenly on purpose -- an evenly spaced series would make the 5th
    percentile agree with almost any interpolation rule, which is exactly the bug this is
    supposed to catch.
    """
    src = tmp_path_factory.mktemp("results")
    for day, base in (("20260517", 0), ("20260518", 7)):
        lines = ["Start (s),End (s),Scientific name,Common name,Confidence"]
        offs = [37, 61, 62, 149, 210, 211, 640, 1811, 1812, 3007, 5400, 7211]
        for i, o in enumerate(offs):                      # a species well above the floor
            lines.append(f"{o + base}.0,{o + base + 3}.0,Turdus migratorius,American Robin,"
                         f"{0.55 + (i % 5) * 0.07:.2f}")
        for o in (500, 2500, 4400):                       # and one that never reaches it
            lines.append(f"{o}.0,{o + 3}.0,Melospiza melodia,Song Sparrow,0.71")
        (src / f"{day}_043000.BirdNET.results.csv").write_text("\n".join(lines) + "\n",
                                                               encoding="utf-8")
    out = tmp_path_factory.mktemp("phen") / "dash.html"
    build_site.main(["--from-analyzer", str(src), "--out", str(out),
                     "--lat", "42.5372", "--lon", "-72.5317", "--tz", "America/New_York"])
    return out


def test_browser_phenology_matches_python(page):
    r = _run(page)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "every field matches" in r.stdout, r.stdout


def test_overview_follows_the_floor_and_matches_the_chapter(page):
    """Two charts, one question, different routes -- so they can drift apart silently.

    The Overview once summed detections straight off the summary, which the detection floor
    does not touch (it only nulls onset), so it reported every species on record while the
    chapter directly beneath it drew the far smaller set that cleared the floor. Neither
    chart looks wrong on its own; only the comparison shows it.
    """
    r = _run(page)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "follows the floor" in r.stdout, r.stdout
    assert "disagree" not in r.stdout, r.stdout


def test_payload_carries_the_raw_times(page):
    """Without `phen` the floor silently reverts to whatever the build used."""
    import re
    src = page.read_text(encoding="utf-8")
    data = json.loads(re.search(r'id="data">(.*?)</script>', src, re.S).group(1))
    assert data.get("phen"), "no per-morning detection times in the payload"
    assert data["meta"].get("phen_species"), "no species index for phen"
    assert data["meta"].get("min_detections") == 5
    # every (day, species) in the summary must be reconstructable
    have = {(d, data["meta"]["phen_species"][int(si)])
            for d, by in data["phen"].items() for si in by}
    want = {(r["date"], r["name"]) for r in data["summary"]}
    assert want <= have, sorted(want - have)[:5]


def test_checker_fails_when_the_two_disagree(page, tmp_path):
    """A guard that cannot fail is not a guard: corrupt the payload, expect a FAIL."""
    import re
    src = page.read_text(encoding="utf-8")
    m = re.search(r'(id="data">)(.*?)(</script>)', src, re.S)
    data = json.loads(m.group(2))
    for r in data["summary"]:
        if r.get("onset") is not None:
            r["onset"] = r["onset"] + 37.0        # a shift no rounding could explain
            break
    broken = tmp_path / "broken.html"
    broken.write_text(src[:m.start(2)] + json.dumps(data) + src[m.end(2):], encoding="utf-8")
    r = _run(broken)
    assert r.returncode != 0 and "FAIL" in r.stdout, r.stdout
