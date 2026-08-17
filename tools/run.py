"""
One entry point for the whole local workflow, driven by `deployments.json`.

Everything here is a thin orchestration layer over the existing tools - it re-types no
science and duplicates no logic, it just supplies each tool the site/recorder facts from
the config so you never pass --lat/--lon/--tz/--recorder by hand (and never get them
wrong, which fails silently rather than loudly).

    python tools/run.py status                 what's configured and what's unprocessed
    python tools/run.py process                inference on NEW recordings, every deployment
    python tools/run.py process --site montague --deployment owl
    python tools/run.py dashboard              rebuild the local click-to-listen pages
    python tools/run.py compare                two recorders at one site, head to head
    python tools/run.py publish [--push]       static JSON for the public site
    python tools/run.py serve                  local server (range requests, so audio seeks)
    python tools/run.py all                    process -> dashboard  (the usual card-dump run)

On Windows, `run.ps1` wraps this so you don't have to name the venv's python.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                                        # noqa: E402
import dawnchorus as dc                              # noqa: E402


def _targets(args):
    return config.deployments(getattr(args, "site", None), getattr(args, "deployment", None))


def cmd_status(args):
    print(config.describe())
    print()
    import track
    for d in _targets(args):
        if not d.audio.exists():
            print(f"{d.label:<18} audio folder missing: {d.audio}")
            continue
        audio = track.scan(str(d.audio))
        man = track.load_manifest(str(d.manifest))
        new, changed, gone = track.diff(audio, man)
        done = len(man.get("files", {}))
        print(f"{d.label:<18} {len(audio):>3} recordings, {done:>3} processed, "
              f"{len(new):>3} new" + (f", {len(changed)} changed" if changed else "")
              + (f", {len(gone)} missing" if gone else ""))


def cmd_process(args):
    import track
    for d in _targets(args):
        if not d.audio.exists():
            print(f"[skip] {d.label}: no folder {d.audio}")
            continue
        print(f"\n=== {d.label}  ({d.recorder}) ===")
        argv = ["process", "--audio", str(d.audio), "--recorder", d.recorder,
                "--lat", str(d.lat), "--lon", str(d.lon), "--tz", d.tz,
                "--capture-conf", str(args.capture_conf),
                "--min-confidence", str(args.min_confidence),
                "--engine", args.engine]
        argv += ["--overlap", str(args.overlap)]
        # Only pin a week when asked. Left alone, the engine derives it per FILE, which
        # matters across a month boundary: pinning one week for a batch spanning Aug 6-12
        # applies late-July's species prior to recordings that are a week later.
        if args.week is not None:
            argv += ["--week", str(args.week)]
        if args.reprocess:
            argv += ["--reprocess"]
        for n in (getattr(args, "only", None) or []):
            argv += ["--only", n]
        track.main(argv)


def cmd_dashboard(args):
    import build_site
    for d in _targets(args):
        if not d.results.exists():
            print(f"[skip] {d.label}: nothing analysed yet ({d.results})")
            continue
        out = ROOT / "site" / f"dashboard-{d.key}.html"
        print(f"\n=== dashboard {d.label} -> {out.name} ===")
        build_site.main(["--from-analyzer", str(d.results), "--audio", str(d.audio),
                         "--site", d.site,
                         "--recorder", d.recorder, "--audio-url-base", f"../{d.audio.name}",
                         "--lat", str(d.lat), "--lon", str(d.lon), "--tz", d.tz,
                         "--min-confidence", str(args.min_confidence), "--out", str(out)])


def cmd_compare(args):
    import compare_recorders
    ds = config.deployments(args.site)
    if len(ds) < 2:
        sys.exit("compare needs two deployments at one site; "
                 f"{ds[0].site if ds else '?'} has {len(ds)}")
    a = config.primary(ds[0].site)
    b = next(d for d in ds if d.key != a.key)
    print(f"comparing {a.label} (reference) vs {b.label}\n")
    compare_recorders.main([
        "--recorder-a", a.recorder, "--results-a", str(a.results), "--audio-a", str(a.audio),
        "--recorder-b", b.recorder, "--results-b", str(b.results), "--audio-b", str(b.audio),
        "--lat", str(a.lat), "--lon", str(a.lon), "--tz", a.tz,
        "--min-confidence", str(args.min_confidence), "--out", str(ROOT / "compare")])


def cmd_publish(args):
    import publish
    d = config.primary(args.site) if args.site else config.primary(
        next(iter(config.load()["sites"])))
    if args.deployment:
        d = config.deployment(d.site, args.deployment)
    print(f"publishing {d.label} as site '{d.site}'")
    argv = ["--slug", d.site, "--name", d.name, "--from-analyzer", str(d.results),
            "--lat", str(d.lat), "--lon", str(d.lon), "--tz", d.tz,
            "--recorder", d.recorder, "--min-confidence", str(args.min_confidence)]
    if d.unit:
        argv += ["--unit", d.unit]
    if args.push:
        argv += ["--push"]
    publish.main(argv)


def cmd_serve(args):
    import serve
    serve.main(["--port", str(args.port)])


def cmd_webapp(args):
    import webapp
    webapp.main(["--port", str(args.port)])


def cmd_all(args):
    cmd_process(args)
    cmd_dashboard(args)


def main(argv=None):
    p = argparse.ArgumentParser(description="Dawn-chorus local workflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, site=True, conf=True):
        if site:
            sp.add_argument("--site", default=None, help="site slug (default: all)")
            sp.add_argument("--deployment", default=None, help="deployment key, e.g. sm / owl")
        if conf:
            sp.add_argument("--min-confidence", dest="min_confidence", type=float,
                            default=dc.CHART_MIN_CONFIDENCE)
        return sp

    common(sub.add_parser("status", help="what's configured and what's unprocessed"))

    for name, help_ in (("process", "run inference on NEW recordings"),
                        ("all", "process, then rebuild the dashboards")):
        sp = common(sub.add_parser(name, help=help_))
        sp.add_argument("--capture-conf", dest="capture_conf", type=float, default=0.25,
                        help="what BirdNET logs (keep low; the dashboard filters higher)")
        sp.add_argument("--week", type=int, default=None,
                        help="BirdNET week 1-48 (default: inferred from the recording dates)")
        sp.add_argument("--overlap", type=float, default=0.0,
                        help="seconds of overlap between 3s windows (0-2.9). Higher catches "
                             "calls that straddle a window boundary, at ~1.2x runtime per "
                             "0.5s and proportionally more detections -- so change it for the "
                             "WHOLE archive at once or n stops being comparable.")
        sp.add_argument("--engine", default="auto", choices=["auto", "litert", "analyzer"])
        sp.add_argument("--only", action="append", default=None, metavar="NAME",
                        help="process exactly these recordings (repeatable)")
        sp.add_argument("--reprocess", action="store_true", help="redo every recording")

    common(sub.add_parser("dashboard", help="rebuild the local click-to-listen pages"))

    sp = common(sub.add_parser("compare", help="two recorders at one site, head to head"),
                site=False)
    sp.add_argument("--site", default=None)
    sp.add_argument("--deployment", default=None)

    sp = common(sub.add_parser("publish", help="regenerate the public site's JSON"))
    sp.add_argument("--push", action="store_true", help="git push (triggers the deploy)")

    sp = sub.add_parser("serve", help="local server for the dashboards")
    sp.add_argument("--port", type=int, default=8000)

    sp = sub.add_parser("webapp", help="local control panel: click to run models")
    sp.add_argument("--port", type=int, default=8765)

    args = p.parse_args(argv)
    {"status": cmd_status, "process": cmd_process, "dashboard": cmd_dashboard,
     "compare": cmd_compare, "publish": cmd_publish, "serve": cmd_serve,
     "webapp": cmd_webapp, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
