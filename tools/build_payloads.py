"""
Generate the STATIC per-site JSON the $0 hosted viewer reads.

The browser never queries a database — it just needs the finished dashboard payload. So we
precompute each site's payload here (offline) and write plain files:

    site/data/sites.json      the site index (drives the picker)
    site/data/<slug>.json     one precomputed payload per site

`build_site.py --emit-viewer` (static mode) points the viewer at these, and your existing
S3/CloudFront deploy serves them for free. Recordings never leave the machine — only the
detections' phenology goes up. Re-running for a slug updates that site and its index entry.

    python tools/build_payloads.py --slug montague --name "Montague, MA" \
        --from-analyzer data/results --lat 42.53 --lon -72.53 --tz America/New_York

Same payload contract as the FastAPI server (server/payload.build_payload), so switching to a
live API later is a one-line viewer change, not a rebuild.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))  # reuse the builder

import dawnchorus as dc
from payload import build_payload  # noqa: E402  (server/payload.py)


def main(argv=None):
    p = argparse.ArgumentParser(description="Write static per-site JSON for the hosted viewer")
    p.add_argument("--slug", required=True, help="short id used in the URL and filename")
    p.add_argument("--name", required=True, help="display name shown in the site picker")
    p.add_argument("--from-analyzer", dest="from_analyzer", required=True,
                   help="folder of BirdNET-Analyzer result tables for this site")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--recorder", default=None,
                   help="recorder profile id for this site (e.g. song-meter-micro-2, owl-sense); "
                        "supplies the filename convention + clock zone and tags the detections")
    p.add_argument("--unit", default=None, help="serial of the physical box, e.g. 2MM43813")
    p.add_argument("--file-tz", dest="file_tz", default=None,
                   help="tz stamped in filenames if not station-local; overrides the profile")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="confidence floor for the charts")
    p.add_argument("--label-min-confidence", dest="label_min_conf", type=float, default=0.25,
                   help="lower floor for spectrogram labels")
    p.add_argument("--out-dir", default="site/data", help="where to write the JSON (default site/data)")
    args = p.parse_args(argv)

    if args.recorder:
        from dawnchorus import recorders as rec
        names = [f.name for f in Path(args.from_analyzer).rglob("*") if f.is_file()]
        print(rec.describe(args.recorder, names))
    det = dc.load_birdnet_analyzer(args.from_analyzer, min_confidence=0.0, latitude=args.lat,
                                   longitude=args.lon, tz=args.tz, file_tz=args.file_tz,
                                   recorder=args.recorder)
    # Curated exclusions for this site, reported rather than applied silently.
    import config as _cfg
    det, excl_notes = _cfg.apply_exclusions(det, args.slug)
    for n in excl_notes:
        print(f"[exclude] {n['date']}: dropped {n['removed']:,} detections of "
              f"{', '.join(n['species'])}")
    payload = build_payload(det, args.lat, args.lon, args.tz,
                            args.min_confidence, args.label_min_conf)
    payload["meta"]["exclusions"] = excl_notes

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.json").write_text(json.dumps(payload), encoding="utf-8")

    mornings = payload["meta"]["mornings"]
    entry = {"slug": args.slug, "name": args.name, "lat": args.lat, "lon": args.lon,
             "tz": args.tz, "n_detections": int(len(det)),
             "first": mornings[0] if mornings else None,
             "last": mornings[-1] if mornings else None,
             "recorder": args.recorder, "unit": args.unit}
    idx_path = out / "sites.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    idx = [s for s in idx if s.get("slug") != args.slug] + [entry]
    idx.sort(key=lambda s: s["slug"])
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")

    m = payload["meta"]
    print(f"wrote {out / (args.slug + '.json')}  ({m['n_detections']:,} charted detections, "
          f"{m['n_species']} species, {len(mornings)} mornings)")
    print(f"index {idx_path} now lists {len(idx)} site(s): {', '.join(s['slug'] for s in idx)}")


if __name__ == "__main__":
    main()
