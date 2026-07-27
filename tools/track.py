"""
File tracking + incremental processing for dawn-chorus recordings.

Keeps a manifest of which recordings have already been run through BirdNET, so you can drop
a new card's WAVs into the audio folder and process only the *new* ones on request.

    python tools/track.py status  --audio data
    python tools/track.py process --audio data --lat 42.53 --lon -72.53 \
        --tz America/New_York --rebuild-site

`status`  reports how many recordings are tracked, which are new/changed/missing.
`process` runs BirdNET-Analyzer on the new/changed recordings, records them in the manifest,
          and (with --rebuild-site) regenerates the dashboard.

Manifest defaults to <audio>/manifest.json. Requires `birdnet-analyzer` for `process`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import build_site  # sibling module in tools/

MANIFEST_NAME = "manifest.json"


def scan(audio_dir):
    out = {}
    for p in sorted(Path(audio_dir).glob("*.wav")):
        st = p.stat()
        out[p.name] = {"size": st.st_size, "mtime": round(st.st_mtime, 1)}
    return out


def load_manifest(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"files": {}, "updated": None}


def diff(audio, manifest):
    files = manifest.get("files", {})
    new = [n for n in audio if n not in files]
    changed = [n for n in audio if n in files and files[n].get("size") != audio[n]["size"]]
    gone = [n for n in files if n not in audio]
    return new, changed, gone


def infer_week(names):
    days = []
    for n in names:
        m = re.search(r"(\d{8})_\d{6}", n)
        if m:
            days.append(date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])))
    if not days:
        return -1
    days.sort()
    mid = days[len(days) // 2]
    return (mid.month - 1) * 4 + min((mid.day - 1) // 7 + 1, 4)


def result_csv(results_dir, wav_name):
    hits = sorted(Path(results_dir).glob(f"{Path(wav_name).stem}*.csv"))
    return hits[0] if hits else None


def count_detections(csv_path):
    try:
        with open(csv_path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return None


def cmd_status(args):
    audio = scan(args.audio)
    man = load_manifest(args.manifest)
    new, changed, gone = diff(audio, man)
    tracked = man.get("files", {})
    total_det = sum(v.get("detections") or 0 for v in tracked.values())
    print(f"audio folder : {len(audio)} recording(s)  ({args.audio})")
    print(f"tracked      : {len(tracked)} processed, {total_det:,} detections  ({args.manifest})")
    print(f"new          : {len(new)}" + (("  " + ", ".join(new[:6]) + (" …" if len(new) > 6 else "")) if new else ""))
    if changed:
        print(f"changed      : {len(changed)}  {', '.join(changed)}")
    if gone:
        print(f"missing      : {len(gone)}  (in manifest, file gone)")
    if not new and not changed:
        print("up to date — nothing to process.")


def cmd_process(args):
    audio = scan(args.audio)
    man = load_manifest(args.manifest)
    new, changed, gone = diff(audio, man)
    todo = new + changed
    results = args.results or str(Path(args.audio) / "results")

    if not todo:
        print("nothing new to process.")
    else:
        week = args.week if args.week is not None else infer_week(todo)
        threads = args.threads or max(2, os.cpu_count() or 4)
        print(f"[analyze] {len(todo)} new/changed recording(s), week {week}, capture >= {args.capture_conf}")
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", args.audio, "--output", results,
               "--lat", str(args.lat), "--lon", str(args.lon), "--week", str(week),
               "--min_conf", str(args.capture_conf), "--rtype", "csv",
               "--threads", str(threads), "--skip_existing_results"]
        subprocess.run(cmd, check=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name in todo:
            csv = result_csv(results, name)
            man.setdefault("files", {})[name] = {**audio[name], "processed_at": stamp,
                                                 "detections": count_detections(csv) if csv else None}
        man["updated"] = stamp
        Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"[manifest] {len(man['files'])} recording(s) tracked -> {args.manifest}")

    if args.rebuild_site:
        print(f"[site] rebuilding {args.out_site} (counting >= {args.min_confidence})")
        data = build_site.build_data(analyzer_path=results, lat=args.lat, lon=args.lon, tz=args.tz,
                                     min_conf=args.min_confidence, file_tz=args.file_tz, audio_dir=args.audio)
        out = Path(args.out_site)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_site.render_html(data), encoding="utf-8")
        m = data["meta"]
        print(f"[site] {m['n_detections']:,} detections, {m['n_species']} species, {len(m['mornings'])} mornings")


def main(argv=None):
    p = argparse.ArgumentParser(description="Track recordings and process new ones on request")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="report new / tracked / missing recordings")
    st.add_argument("--audio", required=True)
    st.add_argument("--manifest", default=None)

    pr = sub.add_parser("process", help="analyze new recordings and update the manifest")
    pr.add_argument("--audio", required=True)
    pr.add_argument("--lat", type=float, required=True)
    pr.add_argument("--lon", type=float, required=True)
    pr.add_argument("--tz", required=True)
    pr.add_argument("--file-tz", dest="file_tz", default=None)
    pr.add_argument("--week", type=int, default=None)
    pr.add_argument("--capture-conf", type=float, default=0.25)
    pr.add_argument("--min-confidence", type=float, default=0.50)
    pr.add_argument("--threads", type=int, default=0)
    pr.add_argument("--results", default=None)
    pr.add_argument("--manifest", default=None)
    pr.add_argument("--rebuild-site", action="store_true")
    pr.add_argument("--out-site", default="site/index.html")

    args = p.parse_args(argv)
    if not getattr(args, "manifest", None):
        args.manifest = str(Path(args.audio) / MANIFEST_NAME)
    (cmd_status if args.cmd == "status" else cmd_process)(args)


if __name__ == "__main__":
    main()
