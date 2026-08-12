"""The generated dashboard's JavaScript must parse.

build_site.py assembles a ~35 KB inline script by string substitution, so a Python-side
change can emit a page whose JS is syntactically broken. Nothing else in the build, the
suite or the deploy would catch it -- the browser just renders a blank panel. This caught
a real duplicate `const` before it shipped.

Skipped when node is unavailable, so a contributor without it still gets a green suite;
CI installs node so the check genuinely runs there.
"""
from __future__ import annotations

import pytest

import build_viewer
import check_js

pytestmark = pytest.mark.skipif(not check_js.have_node(),
                                reason="node is not installed; cannot parse the emitted JS")


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    """The public viewer, built from the same template as the local dashboards."""
    out = tmp_path_factory.mktemp("viewer") / "index.html"
    build_viewer.main(["--out", str(out)])
    return out


def test_viewer_javascript_parses(viewer, tmp_path):
    assert check_js.check([viewer], keep_dir=tmp_path) == []


def test_api_mode_javascript_parses(tmp_path):
    """The --api-base variant injects a different bootstrap, so it needs its own check."""
    out = tmp_path / "api.html"
    build_viewer.main(["--out", str(out), "--api-base", "https://example.org"])
    assert check_js.check([out], keep_dir=tmp_path) == []


def test_checker_actually_fails_on_broken_javascript(tmp_path):
    """A guard that cannot fail is not a guard."""
    bad = tmp_path / "bad.html"
    bad.write_text("<script>\nconst a = 1;\nconst a = 2;\nfunction ( {\n</script>",
                   encoding="utf-8")
    failures = check_js.check([bad], keep_dir=tmp_path)
    assert failures and "bad.html" in failures[0]


def test_missing_script_is_reported(tmp_path):
    """If the template stops emitting a script at all, say so rather than pass silently."""
    empty = tmp_path / "empty.html"
    empty.write_text("<html><body>no script here</body></html>", encoding="utf-8")
    assert any("no inline script" in f for f in check_js.check([empty], keep_dir=tmp_path))
