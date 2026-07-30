"""
File tracking + incremental processing for dawn-chorus recordings.

Keeps a manifest of which recordings have already been run through BirdNET, so you can drop
a new card's WAVs into the audio folder and process only the *new* ones on request.

    python tools/track.py status  --audio data
    python tools/track.py process --audio data --lat 42.53 --lon -72.53 \
        --tz America/New_York --rebuild-site
    python tools/track.py upload  --audio data --slug montague --api-key <site-key>

`status`  reports how many recordings are tracked, which are new/changed/missing.
`process` runs BirdNET-Analyzer on the new/changed recordings, records them in the manifest,
          and (with --rebuild-site) regenerates the dashboard.
`upload`  sends detections for processed recordings to the shared API (the WAVs stay local);
          each file is marked uploaded so re-runs only send what's new.

Manifest defaults to <audio>/manifest.json. Requires `birdnet-analyzer` for `process` and a
running dawn-chorus API (see server/) plus the site's key for `upload`.
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


def cmd_list(args):
    audio = scan(args.audio)
    files = load_manifest(args.manifest).get("files", {})
    print(f"{'recording':42} {'date':11} {'model':12} {'det':>5}  status")
    print("-" * 82)
    for name in sorted(audio):
        m = re.search(r"(\d{8})_\d{6}", name)
        d = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}" if m else "?"
        info = files.get(name)
        if info:
            det = info.get("detections")
            print(f"{name:42} {d:11} {(info.get('model') or 'default'):12} "
                  f"{('?' if det is None else det):>5}  ✓ {str(info.get('processed_at',''))[:10]}")
        else:
            print(f"{name:42} {d:11} {'-':12} {'-':>5}  NEW")
    for name in sorted(files):
        if name not in audio:
            print(f"{name:42} {'':11} {'':12} {'':>5}  MISSING (file gone)")


def cmd_process(args):
    audio = scan(args.audio)
    man = load_manifest(args.manifest)
    new, changed, gone = diff(audio, man)
    todo = list(audio) if args.reprocess else (new + changed)
    results = args.results or str(Path(args.audio) / "results")
    model = Path(args.classifier).stem if args.classifier else "default"

    if not todo:
        print("nothing new to process.")
    else:
        week = args.week if args.week is not None else infer_week(todo)
        threads = args.threads or max(2, os.cpu_count() or 4)
        print(f"[analyze] {len(todo)} recording(s), model {model}, week {week}, capture >= {args.capture_conf}")
        cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", args.audio, "--output", results,
               "--lat", str(args.lat), "--lon", str(args.lon), "--week", str(week),
               "--min_conf", str(args.capture_conf), "--overlap", str(args.overlap),
               "--sensitivity", str(args.sensitivity), "--rtype", "csv", "--threads", str(threads)]
        if args.classifier:
            cmd += ["-c", args.classifier]
        if not args.reprocess:
            cmd.append("--skip_existing_results")
        subprocess.run(cmd, check=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name in todo:
            csv = result_csv(results, name)
            man.setdefault("files", {})[name] = {**audio[name], "processed_at": stamp, "model": model,
                                                 "detections": count_detections(csv) if csv else None}
        man["updated"] = stamp
        Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"[manifest] {len(man['files'])} recording(s) tracked -> {args.manifest}")

    if args.rebuild_site:
        print(f"[site] rebuilding {args.out_site} (counting >= {args.min_confidence})")
        data = build_site.build_data(analyzer_path=results, lat=args.lat, lon=args.lon, tz=args.tz,
                                     min_conf=args.min_confidence, file_tz=args.file_tz, audio_dir=args.audio,
                                     label_min_conf=args.label_min_conf)
        out = Path(args.out_site)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_site.render_html(data), encoding="utf-8")
        m = data["meta"]
        print(f"[site] {m['n_detections']:,} detections, {m['n_species']} species, {len(m['mornings'])} mornings")


def _post(url, key, chunk):
    """POST one batch of detections; return the server's inserted count."""
    import urllib.error
    import urllib.request
    body = json.dumps(chunk).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()).get("inserted", len(chunk))
    except urllib.error.HTTPError as e:
        sys.exit(f"upload failed ({e.code}) at {url}: {e.read().decode('utf-8', 'replace')[:200]}")
    except urllib.error.URLError as e:
        sys.exit(f"can't reach the API at {url}: {e.reason}")


def cmd_upload(args):
    """Push detections for processed-but-not-yet-uploaded recordings to the API.

    Recordings stay local; only detections (dt, species, confidence) are sent. Each file is
    marked uploaded in the manifest so re-running only sends new ones. The server keeps every
    detection and filters by confidence at query time, so we upload at a low floor.
    """
    import pandas as pd

    import dawnchorus as dc
    audio = scan(args.audio)
    man = load_manifest(args.manifest)
    files = man.get("files", {})
    results = args.results or str(Path(args.audio) / "results")

    todo = [n for n in sorted(audio) if files.get(n)                       # processed
            and (args.reupload or not files[n].get("uploaded_at"))]        # and not yet sent
    if not todo:
        print("nothing to upload — all processed recordings already sent (use --reupload to force).")
        return
    key = args.api_key or os.environ.get("DAWNCHORUS_API_KEY")
    if not key:
        sys.exit("no API key — pass --api-key or set DAWNCHORUS_API_KEY")

    url = args.api_base.rstrip("/") + f"/sites/{args.slug}/detections"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    print(f"[upload] {len(todo)} recording(s) -> {url}  (floor >= {args.upload_min_confidence})")
    for name in todo:
        csv = result_csv(results, name)
        dets = []
        if csv:
            try:
                det = dc.load_birdnet_analyzer(csv, min_confidence=args.upload_min_confidence,
                                               tz=args.tz, file_tz=args.file_tz)
                for r in det.itertuples():
                    d = pd.Timestamp(r.datetime).to_pydatetime()
                    if d.tzinfo is not None:
                        d = d.replace(tzinfo=None)                          # naive station-local
                    dets.append({"dt": d.isoformat(), "scientific_name": str(r.scientific_name),
                                 "common_name": str(r.common_name), "confidence": float(r.confidence)})
            except ValueError:
                dets = []                                                   # no detections in this file
        sent = sum(_post(url, key, dets[i:i + args.batch]) for i in range(0, len(dets), args.batch))
        files.setdefault(name, {}).update(uploaded_at=stamp, uploaded=sent)
        total += sent
        print(f"  {name:42} {sent:>5} detections")
    man["files"] = files
    Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[upload] {total:,} detections sent for site '{args.slug}'")


def main(argv=None):
    p = argparse.ArgumentParser(description="Track recordings and process new ones on request")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="report new / tracked / missing recordings")
    st.add_argument("--audio", required=True)
    st.add_argument("--manifest", default=None)

    ls = sub.add_parser("list", help="per-recording table: processed?, detections, model")
    ls.add_argument("--audio", required=True)
    ls.add_argument("--manifest", default=None)

    pr = sub.add_parser("process", help="analyze new (or all) recordings and update the manifest")
    pr.add_argument("--audio", required=True)
    pr.add_argument("--lat", type=float, required=True)
    pr.add_argument("--lon", type=float, required=True)
    pr.add_argument("--tz", required=True)
    pr.add_argument("--file-tz", dest="file_tz", default=None)
    pr.add_argument("--week", type=int, default=None)
    pr.add_argument("--capture-conf", type=float, default=0.25)
    pr.add_argument("--min-confidence", type=float, default=0.50)
    pr.add_argument("--label-min-confidence", dest="label_min_conf", type=float, default=0.25,
                    help="confidence floor for spectrogram labels (charts use --min-confidence)")
    pr.add_argument("--overlap", type=float, default=0.0,
                    help="0-2.9s window overlap; higher = near-continuous coverage, more detections (slower)")
    pr.add_argument("--sensitivity", type=float, default=1.0, help="0.5-1.5; higher = more eager detections")
    pr.add_argument("--threads", type=int, default=0)
    pr.add_argument("--results", default=None)
    pr.add_argument("--manifest", default=None)
    pr.add_argument("--rebuild-site", action="store_true")
    pr.add_argument("--out-site", default="site/index.html")
    pr.add_argument("--classifier", default=None, help="path to a custom BirdNET classifier (default model if omitted)")
    pr.add_argument("--reprocess", action="store_true", help="re-run every recording, not just new/changed")

    up = sub.add_parser("upload", help="push detections for processed recordings to the API")
    up.add_argument("--audio", required=True)
    up.add_argument("--slug", required=True, help="site slug on the server")
    up.add_argument("--api-base", default="http://127.0.0.1:8001")
    up.add_argument("--api-key", default=None, help="site upload key (or env DAWNCHORUS_API_KEY)")
    up.add_argument("--tz", default=None, help="station tz; required only with --file-tz")
    up.add_argument("--file-tz", dest="file_tz", default=None,
                    help="tz stamped in filenames if not station-local (e.g. UTC for AudioMoth)")
    up.add_argument("--upload-min-confidence", dest="upload_min_confidence", type=float, default=0.0,
                    help="floor for uploaded detections; the server filters higher at query time")
    up.add_argument("--results", default=None)
    up.add_argument("--manifest", default=None)
    up.add_argument("--batch", type=int, default=2000, help="detections per POST request")
    up.add_argument("--reupload", action="store_true",
                    help="re-send already-uploaded recordings (appends on the server)")

    args = p.parse_args(argv)
    if not getattr(args, "manifest", None):
        args.manifest = str(Path(args.audio) / MANIFEST_NAME)
    {"status": cmd_status, "list": cmd_list, "process": cmd_process,
     "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
