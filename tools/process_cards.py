"""
One command: raw recordings -> BirdNET-Analyzer -> dashboard site.

    python tools/process_cards.py --audio data --lat 42.53 --lon -72.53 \
        --tz America/New_York --out-site site/index.html

Runs BirdNET-Analyzer over a folder of recordings (writing result CSVs), then rebuilds the
dashboard from them. Point it at a fresh card dump and you get an updated site.

Requires `birdnet-analyzer` installed in the same environment (`pip install birdnet-analyzer`).
Recording filenames must carry a YYYYMMDD_HHMMSS timestamp (AudioMoth / Song Meter). The
recorder clock is assumed to be station-local unless you pass --file-tz (e.g. UTC).

Two confidence thresholds, by design (capture low, analyse higher):
  --capture-conf   what BirdNET logs           (default 0.25)
  --min-confidence what the dashboard counts    (default 0.50)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import build_site  # sibling module in tools/


def infer_week(audio_dir: str) -> int:
    """BirdNET week-of-year [1..48] from the median recording date; -1 if unknown."""
    days = []
    for p in Path(audio_dir).glob("*.wav"):
        m = re.search(r"(\d{8})_\d{6}", p.name)
        if m:
            y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
            days.append(date(y, mo, d))
    if not days:
        return -1
    days.sort()
    mid = days[len(days) // 2]
    return (mid.month - 1) * 4 + min((mid.day - 1) // 7 + 1, 4)


def main(argv=None):
    p = argparse.ArgumentParser(description="Recordings -> BirdNET-Analyzer -> dashboard")
    p.add_argument("--audio", required=True, help="folder of WAV recordings")
    p.add_argument("--results", default=None, help="result-table folder (default <audio>/results)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--file-tz", dest="file_tz", default=None,
                   help="tz the recording filenames are in (e.g. UTC); omit if local")
    p.add_argument("--week", type=int, default=None, help="BirdNET week 1..48 (default: inferred)")
    p.add_argument("--capture-conf", type=float, default=0.25)
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--overlap", type=float, default=0.0, help="0-2.9s window overlap (higher = more detections, slower)")
    p.add_argument("--sensitivity", type=float, default=1.0, help="0.5-1.5; higher = more eager detections")
    p.add_argument("--threads", type=int, default=0, help="0 = auto (CPU count)")
    p.add_argument("--out-site", default="site/dashboard-local.html")
    p.add_argument("--skip-analyze", action="store_true",
                   help="reuse existing result CSVs; only rebuild the site")
    args = p.parse_args(argv)

    results = args.results or str(Path(args.audio) / "results")
    week = args.week if args.week is not None else infer_week(args.audio)

    if not args.skip_analyze:
        import os
        threads = args.threads or max(2, os.cpu_count() or 4)
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", args.audio,
               "--output", results, "--lat", str(args.lat), "--lon", str(args.lon),
               "--week", str(week), "--min_conf", str(args.capture_conf),
               "--overlap", str(args.overlap), "--sensitivity", str(args.sensitivity),
               "--rtype", "csv", "--threads", str(threads), "--skip_existing_results"]
        print(f"[1/2] BirdNET-Analyzer -> {results}  (week {week}, capture >= {args.capture_conf})")
        subprocess.run(cmd, check=True)
    else:
        print(f"[1/2] skipping analysis; reusing {results}")

    print(f"[2/2] building dashboard -> {args.out_site}  (counting >= {args.min_confidence})")
    data = build_site.build_data(analyzer_path=results, lat=args.lat, lon=args.lon,
                                 tz=args.tz, min_conf=args.min_confidence, file_tz=args.file_tz,
                                 audio_dir=args.audio)
    out = Path(args.out_site)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_site.render_html(data), encoding="utf-8")
    m = data["meta"]
    print(f"done: {m['n_detections']:,} detections, {m['n_species']} species, "
          f"{len(m['mornings'])} mornings -> {out}")


if __name__ == "__main__":
    main()
