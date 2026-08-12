"""
Syntax-check the JavaScript embedded in a generated page.

The dashboard is one ~35 KB inline script built by string substitution in build_site.py.
Python happily emits a page whose JavaScript does not parse, the browser then renders a
blank panel, and nothing in the build, the tests or the deploy notices. That is not
hypothetical: a `const DAWN` added for the clock-time axis collided with the one the audio
player already declared, which would have blanked every chart on every page.

`node --check` is the cheapest possible guard. It parses without executing, so it needs no
DOM, no Plot, and no data.

    python tools/check_js.py site/index.html site/dashboard-sm.html

Exits non-zero if any block fails. Skips the `application/json` data island, which is data
rather than code.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Any <script> that is not the JSON data island.
SCRIPT = re.compile(r'<script(?![^>]*type=["\']application/json)[^>]*>(.*?)</script>', re.S)
MIN_CHARS = 40                      # ignore one-liner shims; nothing to get wrong


def have_node() -> bool:
    return shutil.which("node") is not None


def check(paths, keep_dir: Path | None = None) -> list[str]:
    """Return a list of human-readable failures; empty means every block parsed."""
    if not have_node():
        raise RuntimeError("node is not on PATH; cannot syntax-check the emitted JavaScript")
    out = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="dc_js_"))
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for p in map(Path, paths):
        if not p.exists():
            failures.append(f"{p}: missing")
            continue
        blocks = [b for b in SCRIPT.findall(p.read_text(encoding="utf-8"))
                  if len(b.strip()) > MIN_CHARS]
        if not blocks:
            failures.append(f"{p}: no inline script found (did the template change?)")
            continue
        for i, b in enumerate(blocks):
            f = out / f"{p.stem}_{i}.js"
            f.write_text(b, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            if r.returncode != 0:
                detail = "\n".join((r.stderr or "").strip().splitlines()[:6])
                failures.append(f"{p.name} block {i} ({len(b):,} chars):\n{detail}")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description="node --check the inline JS of generated pages")
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--keep", default=None, help="write the extracted blocks here for debugging")
    args = ap.parse_args(argv)

    if not have_node():
        print("node not found on PATH - skipping the JavaScript syntax check")
        return 0
    failures = check(args.pages, args.keep)
    for f in failures:
        print(f"FAIL {f}")
    if failures:
        print(f"\n{len(failures)} block(s) failed to parse")
        return 1
    print(f"all inline JavaScript parses ({len(args.pages)} page(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
