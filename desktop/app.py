"""
Dawn Chorus desktop app — record folder -> BirdNET -> publishable site JSON, no Python install.

Bundled with PyInstaller (LiteRT, not TensorFlow, so it stays small). A contributor picks their
recordings folder, enters their site name + coordinates, and clicks Run; the app produces a
small <slug>.json to publish (open a PR / send it to the maintainer). Recordings never leave
the machine — only detection times are written.

    python desktop/app.py                       # GUI
    python desktop/app.py --cli --folder data --slug montague --name "North St, Montague, MA" \
        --lat 42.537278 --lon -72.531694 --tz America/New_York --out out/

The inference core is birdnet_lite (validated byte-for-byte against BirdNET-Analyzer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = Path(getattr(sys, "_MEIPASS", str(HERE)))    # PyInstaller unpack dir when frozen, else desktop/
for p in (str(ROOT), str(ROOT / "server")):         # dawnchorus + build_payload (dev; bundled in .exe)
    if p not in sys.path:
        sys.path.insert(0, p)

import birdnet_lite as bl


def _model_dir() -> Path:
    """Where the three .tflite models live: next to the exe (bundled) or a BirdNET-Analyzer install."""
    if os.environ.get("BIRDNET_MODELS"):
        return Path(os.environ["BIRDNET_MODELS"])
    bundled = BASE / "models"
    if (bundled / "model.tflite").exists() or (bundled / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite").exists():
        return bundled
    import importlib.util
    spec = importlib.util.find_spec("birdnet_analyzer")          # dev fallback
    if spec:
        return Path(os.path.dirname(spec.origin)) / "checkpoints" / "V2.4"
    raise FileNotFoundError("BirdNET models not found; set BIRDNET_MODELS to their folder")


def _models(md: Path):
    def pick(*names):
        for n in names:
            if (md / n).exists():
                return str(md / n)
        raise FileNotFoundError(f"none of {names} under {md}")
    return (pick("model.tflite", "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"),
            pick("labels.txt", "BirdNET_GLOBAL_6K_V2.4_Labels.txt"),
            pick("mdata.tflite", "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"))


def run_pipeline(folder, slug, name, lat, lon, tz, out_dir, min_conf=0.5,
                 label_min_conf=0.25, capture_conf=0.25, log=print, recorder=None):
    """Analyze a folder and write <slug>.json + update sites.json. Returns a summary dict.

    `recorder` is a profile id from `dawnchorus.recorders`. It decides how filenames encode
    the recording start and whether that clock is UTC — get it wrong and every solar time is
    off by the station's UTC offset, which looks like a plausible answer rather than an error.
    """
    import pandas as pd

    from dawnchorus import recorders as rec
    from payload import build_payload

    prof = rec.get(recorder)
    ts_regex = ts_format = None
    if prof is not None:
        import glob
        import os
        names = [os.path.basename(w) for w in
                 glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.WAV"))]
        prof = prof.resolve(names)
        ts_regex, ts_format = prof.timestamp_spec()
        log(rec.describe(recorder, names))

    model_p, labels_p, mdata_p = _models(_model_dir())
    log("loading BirdNET (LiteRT)…")
    interp, labels = bl.load(model_p, labels_p)
    meta = bl.load_meta(mdata_p)
    log("analyzing recordings (they stay on this machine)…")
    dets = bl.analyze_folder(folder, interp, labels, lat, lon, meta=meta, min_conf=capture_conf,
                             progress=lambda k, n, nm: log(f"  [{k}/{n}] {nm}"),
                             ts_regex=ts_regex, ts_format=ts_format)
    if not dets:
        raise RuntimeError(f"no detections found in {folder} — is it the right folder of .wav files?")
    df = pd.DataFrame(dets)[["datetime", "scientific_name", "common_name", "confidence"]]

    # Recorders that stamp filenames in UTC (AudioMoth's default) must be converted to
    # station-local before any solar anchoring happens.
    if prof is not None and prof.file_tz:
        log(f"converting filename times from {prof.file_tz} to {tz}…")
        df["datetime"] = (pd.to_datetime(df["datetime"])
                          .dt.tz_localize(prof.file_tz, ambiguous="NaT", nonexistent="shift_forward")
                          .dt.tz_convert(tz).dt.tz_localize(None))
        df = df.dropna(subset=["datetime"])
    if prof is not None:
        df["recorder"] = prof.id

    log(f"{len(df)} detections; building the dashboard payload…")
    payload = build_payload(df, lat, lon, tz, min_conf, label_min_conf)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(payload), encoding="utf-8")
    idx_path = out / "sites.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    mornings = payload["meta"]["mornings"]
    entry = {"slug": slug, "name": name, "lat": lat, "lon": lon, "tz": tz, "n_detections": int(len(df)),
             "first": mornings[0] if mornings else None, "last": mornings[-1] if mornings else None,
             "recorder": (prof.id if prof is not None else None)}
    idx = [s for s in idx if s.get("slug") != slug] + [entry]
    idx.sort(key=lambda s: s["slug"])
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return {"detections": int(len(df)), "species": payload["meta"]["n_species"],
            "mornings": len(mornings), "json": str(out / f"{slug}.json")}


# --- GUI ------------------------------------------------------------------------------------
def launch_gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, ttk

    from dawnchorus import recorders as rec

    root = tk.Tk()
    root.title("Dawn Chorus — publish a site")
    root.geometry("640x600")
    pad = {"padx": 10, "pady": 4}
    vars_ = {k: tk.StringVar(value=v) for k, v in
             {"folder": "", "slug": "", "name": "", "lat": "", "lon": "", "tz": "America/New_York",
              "recorder": "generic", "out": str(Path.home() / "dawn-chorus-output")}.items()}

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    rows = [("Recordings folder", "folder", True), ("Output folder", "out", True),
            ("Site id (slug)", "slug", False), ("Site name", "name", False),
            ("Latitude", "lat", False), ("Longitude", "lon", False), ("Time zone", "tz", False)]
    for i, (lbl, key, browse) in enumerate(rows):
        ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=vars_[key], width=48).grid(row=i, column=1, sticky="we", **pad)
        if browse:
            ttk.Button(frm, text="Browse…",
                       command=lambda k=key: vars_[k].set(filedialog.askdirectory() or vars_[k].get())
                       ).grid(row=i, column=2, **pad)

    # Recorder is a dropdown, not free text: picking the wrong one silently shifts every
    # solar time, so the contributor should be choosing from what the code actually knows.
    r_row = len(rows)
    ttk.Label(frm, text="Recorder").grid(row=r_row, column=0, sticky="w", **pad)
    ttk.Combobox(frm, textvariable=vars_["recorder"], width=45, state="readonly",
                 values=sorted(rec.REGISTRY)).grid(row=r_row, column=1, sticky="we", **pad)
    rows = rows + [("Recorder", "recorder", False)]
    frm.columnconfigure(1, weight=1)

    logbox = tk.Text(frm, height=14, wrap="word", state="disabled", bg="#0b0f16", fg="#eef2f8")
    logbox.grid(row=len(rows) + 1, column=0, columnspan=3, sticky="nsew", **pad)
    frm.rowconfigure(len(rows) + 1, weight=1)

    def log(msg):
        logbox.configure(state="normal"); logbox.insert("end", str(msg) + "\n")
        logbox.see("end"); logbox.configure(state="disabled"); root.update_idletasks()

    def run():
        try:
            v = {k: vars_[k].get().strip() for k in vars_}
            s = run_pipeline(v["folder"], v["slug"], v["name"], float(v["lat"]), float(v["lon"]),
                             v["tz"], v["out"], log=log, recorder=v["recorder"] or None)
            log(f"\n✓ Done — {s['detections']} detections, {s['species']} species, {s['mornings']} mornings")
            log(f"Wrote {s['json']}")
            log("To publish: send this file (and sites.json) to the maintainer, or open a pull request.")
        except Exception as e:
            log(f"\n✗ Error: {e}")
        finally:
            btn.configure(state="normal")

    btn = ttk.Button(frm, text="Run", command=lambda: (btn.configure(state="disabled"),
                     threading.Thread(target=run, daemon=True).start()))
    btn.grid(row=len(rows), column=1, sticky="e", **pad)
    root.mainloop()


def main(argv=None):
    p = argparse.ArgumentParser(description="Dawn Chorus desktop app")
    p.add_argument("--cli", action="store_true", help="run headless instead of launching the GUI")
    p.add_argument("--folder"); p.add_argument("--slug"); p.add_argument("--name")
    p.add_argument("--lat", type=float); p.add_argument("--lon", type=float); p.add_argument("--tz")
    p.add_argument("--recorder", default=None,
                   help="recorder profile id (see dawnchorus/recorders.py); sets the filename "
                        "convention and clock zone, and tags the detections")
    p.add_argument("--out", default="dawn-chorus-output")
    p.add_argument("--min-confidence", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.cli:
        s = run_pipeline(args.folder, args.slug, args.name, args.lat, args.lon, args.tz,
                         args.out, min_conf=args.min_confidence, recorder=args.recorder)
        print(json.dumps(s, indent=2))
    else:
        launch_gui()


if __name__ == "__main__":
    main()
