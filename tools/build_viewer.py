"""
Write the public multi-site viewer page (the hosted landing at index.html).

The viewer is a static HTML shell with no data baked in — it fetches the site list and each
site's payload at load time. By default it reads the static JSON produced by
`build_payloads.py` (sites.json + <slug>.json), which your S3/CloudFront serves for free; pass
`--api-base` instead to point it at the live FastAPI server.

    python tools/build_viewer.py                      # -> site/index.html, reads ./data (static, $0)
    python tools/build_viewer.py --api-base https://api.example.org   # -> live API mode

This is the deployed page. The click-to-listen dashboard (build_site.py) is a separate,
git-ignored local tool, since it needs the recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import build_site  # sibling module in tools/


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the public multi-site viewer (index.html)")
    p.add_argument("--out", default="site/index.html")
    p.add_argument("--data-base", default="./data",
                   help="URL/path of the static JSON (sites.json + <slug>.json); default ./data")
    p.add_argument("--api-base", default=None,
                   help="if set, the viewer fetches from this live API instead of static JSON")
    args = p.parse_args(argv)

    if args.api_base:
        html, note = build_site.render_viewer(args.api_base, "api"), f"live API {args.api_base}"
    else:
        html, note = build_site.render_viewer(args.data_base, "static"), f"static JSON at {args.data_base}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  (viewer -> {note})")


if __name__ == "__main__":
    main()
