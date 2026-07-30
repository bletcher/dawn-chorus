"""
One command to publish a site: regenerate its static JSON and (optionally) push it live.

    python tools/publish.py --slug montague --name "North St, Montague, MA" \
        --from-analyzer data/results --lat 42.537278 --lon -72.531694 \
        --tz America/New_York --push

Runs build_payloads (recordings stay local — only detection times go up), commits the changed
`site/data/` files, and with --push pushes to main, which triggers the S3/CloudFront deploy
(live in ~1 minute). Detect new recordings first with:

    python tools/track.py process --audio <dir> --lat .. --lon .. --tz ..
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import build_payloads  # sibling module in tools/


def _git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{(r.stderr or r.stdout).strip()}")
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description="Regenerate a site's static JSON and publish it")
    p.add_argument("--slug", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--from-analyzer", dest="from_analyzer", required=True,
                   help="folder of BirdNET-Analyzer result tables for this site")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--file-tz", dest="file_tz", default=None)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--label-min-confidence", dest="label_min_conf", type=float, default=0.25)
    p.add_argument("--push", action="store_true",
                   help="git push after committing (triggers the deploy); otherwise just commit")
    args = p.parse_args(argv)

    # 1) regenerate the static payload + the picker index (recordings never leave the machine)
    bp = ["--slug", args.slug, "--name", args.name, "--from-analyzer", args.from_analyzer,
          "--lat", str(args.lat), "--lon", str(args.lon), "--tz", args.tz,
          "--min-confidence", str(args.min_confidence),
          "--label-min-confidence", str(args.label_min_conf)]
    if args.file_tz:
        bp += ["--file-tz", args.file_tz]
    build_payloads.main(bp)

    # 2) commit only the site data (nothing else)
    files = ["site/data/sites.json", f"site/data/{args.slug}.json"]
    _git("add", *files)
    if not _git("status", "--porcelain", "site/data", check=False).stdout.strip():
        print("nothing changed — site JSON already up to date.")
        return
    _git("commit", "-m", f"publish: update site '{args.slug}'")
    print(f"committed {', '.join(files)}")

    # 3) push -> the deploy workflow syncs site/ to S3 and invalidates CloudFront
    if args.push:
        _git("push", "origin", "main")
        print("pushed → deploy triggered; live in ~1 min at your /dawn-chorus/ URL")
    else:
        print("commit made. run 'git push origin main' to deploy (or re-run with --push).")


if __name__ == "__main__":
    main()
