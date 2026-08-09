"""
Inference back ends: recordings -> BirdNET-Analyzer-format result tables.

Both engines emit the SAME artefact - one `<stem>.BirdNET.results.csv` per recording, in
BirdNET-Analyzer's column layout - so everything downstream (the manifest,
`load_birdnet_analyzer`, the payload builder, both dashboards, the comparison tools) is
identical whichever one ran. Swapping engines is not supposed to change any number.

  litert    ai-edge-litert + librosa, no TensorFlow (~150 MB). The DEFAULT.
            Wraps desktop/birdnet_lite.py, which was validated byte-for-byte against the
            reference implementation: 480/480 detections, max confidence delta 0.0002.
  analyzer  the upstream `birdnet-analyzer` package (pulls in TensorFlow, ~2 GB). Kept so
            results can be cross-checked against upstream when a discrepancy is suspected.

The engines differ in one operational way: `litert` is handed the exact list of files to
process (so incremental runs do only new recordings), while `analyzer` is pointed at the
folder and relies on its own `--skip_existing_results`.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "desktop")):        # dawnchorus + birdnet_lite
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dawnchorus import recorders as rec              # noqa: E402

CHUNK_SECONDS = 3.0                                   # BirdNET's analysis window
HEADER = ["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence", "File"]


# --- engine selection -------------------------------------------------------------------

def have(name: str) -> bool:
    if name == "litert":
        if importlib.util.find_spec("ai_edge_litert") is None:
            return False
        try:
            model_dir()
            return True
        except FileNotFoundError:
            return False
    if name == "analyzer":
        return importlib.util.find_spec("birdnet_analyzer") is not None
    return False


def resolve(engine: str = "auto") -> str:
    """Pick an engine. 'auto' prefers litert (no TensorFlow) and falls back to analyzer."""
    if engine in ("litert", "analyzer"):
        if not have(engine):
            raise RuntimeError(
                f"engine {engine!r} is not available. "
                + ("Install ai-edge-litert + librosa and make sure the .tflite models are in "
                   "desktop/models/ (or set BIRDNET_MODELS)." if engine == "litert"
                   else "pip install birdnet-analyzer (pulls in TensorFlow)."))
        return engine
    for cand in ("litert", "analyzer"):
        if have(cand):
            return cand
    raise RuntimeError(
        "no inference engine available. Install the run stack:\n"
        "    .venv\\Scripts\\python -m pip install -r requirements-run.txt\n"
        "and make sure BirdNET's .tflite models are in desktop/models/ (or set BIRDNET_MODELS).")


def model_dir() -> Path:
    """Where the three .tflite files live: env override, desktop/models, or an analyzer install."""
    if os.environ.get("BIRDNET_MODELS"):
        return Path(os.environ["BIRDNET_MODELS"])
    bundled = ROOT / "desktop" / "models"
    if (bundled / "model.tflite").exists() or \
       (bundled / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite").exists():
        return bundled
    spec = importlib.util.find_spec("birdnet_analyzer")
    if spec and spec.origin:
        return Path(os.path.dirname(spec.origin)) / "checkpoints" / "V2.4"
    raise FileNotFoundError("BirdNET models not found; set BIRDNET_MODELS to their folder")


def _model_files(md: Path):
    def pick(*names):
        for n in names:
            if (md / n).exists():
                return str(md / n)
        raise FileNotFoundError(f"none of {names} under {md}")
    return (pick("model.tflite", "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"),
            pick("labels.txt", "BirdNET_GLOBAL_6K_V2.4_Labels.txt"),
            pick("mdata.tflite", "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"))


# --- the engines ------------------------------------------------------------------------

def _write_results(path: Path, dets: list[dict], wav: Path) -> None:
    """Write one recording's detections in BirdNET-Analyzer's CSV layout.

    The `File` column carries the source WAV's absolute path because
    `load_birdnet_analyzer` PREFERS it over the result filename when reconstructing
    timestamps - so it has to be right, not decorative.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for d in dets:
            off = float(d["offset_s"])
            w.writerow([f"{off:.1f}", f"{off + CHUNK_SECONDS:.1f}", d["scientific_name"],
                        d["common_name"], f"{float(d['confidence']):.4f}", str(wav.resolve())])


# Worker state. LiteRT interpreters are not picklable and loading one costs a second or
# two, so each pool process loads its own ONCE via the initializer and reuses it.
_W: dict = {}


def _init_worker(model_p, labels_p, mdata_p):
    import birdnet_lite as bl
    _W["bl"] = bl
    _W["interp"], _W["labels"] = bl.load(model_p, labels_p)
    _W["meta"] = bl.load_meta(mdata_p)


def _run_one(task):
    (path, out_dir, lat, lon, week, min_conf, overlap, sensitivity,
     ts_regex, ts_format) = task
    bl = _W["bl"]
    p = Path(path)
    wk = week
    if wk is None:                       # per-file week: the species filter should follow
        start = bl._file_start(p.name, ts_regex, ts_format)   # the recording's own date
        wk = bl.week_from_date(start) if start else -1
    allowed = bl.species_mask(_W["meta"], lat, lon, wk)
    dets = bl.analyze_file(str(p), _W["interp"], _W["labels"], min_conf, overlap,
                           sensitivity, allowed, ts_regex, ts_format)
    csv_path = Path(out_dir) / f"{p.stem}.BirdNET.results.csv"
    _write_results(csv_path, dets, p)
    return str(csv_path), p.name, len(dets)


def _free_gb() -> float | None:
    """Available RAM in GB, or None if we can't tell. No third-party dependency."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1e9
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return None


# Bytes of peak RAM per output sample, measured rather than guessed. Calibrated on 1-hour
# 24 kHz recordings (172.8M samples at BirdNET's 48 kHz): 2 workers ran fine with 10.7 GB
# free, 4 workers raised MemoryError with ~10.6 GB free, which brackets per-worker peak
# between ~2.7 and ~5.3 GB. 16 B/sample puts a 1-hour file at ~2.8 GB, which reproduces
# both observations. The cost is resampy's resampling (int64 index + float work arrays
# over the OUTPUT length), not inference.
BYTES_PER_OUTPUT_SAMPLE = 16
DEFAULT_PEAK_GB = 3.0


def _peak_gb_per_worker(paths) -> float:
    """Peak RAM for the largest recording in `paths`, in GB.

    Scales with recording LENGTH, so a 2-hour Owl file needs roughly twice a 1-hour Song
    Meter file. Sizing the pool by core count instead of this is how you get a MemoryError.
    """
    biggest = 0
    for p in paths:
        try:
            with wave.open(str(p)) as w:
                biggest = max(biggest, w.getnframes() / w.getframerate())
        except Exception:
            continue
    if not biggest:
        return DEFAULT_PEAK_GB
    return max(1.0, biggest * 48000 * BYTES_PER_OUTPUT_SAMPLE / 1e9)


def _plan_jobs(paths, jobs) -> int:
    if jobs:
        return max(1, jobs)
    cap = max(1, min(len(paths), (os.cpu_count() or 4) // 2))
    free, need = _free_gb(), _peak_gb_per_worker(paths)
    if free is None:
        return min(cap, 2)                       # unknown memory: stay conservative
    # 0.75, not 0.5: the estimate is a transient resampling peak and workers rarely hit it
    # at the same moment. A worker that does run out is retried serially, so an occasional
    # over-commit costs time, not the run.
    return max(1, min(cap, int(free * 0.75 // need)))


def _litert(paths, out_dir, lat, lon, week, min_conf, recorder, overlap, sensitivity,
            progress, jobs=0):
    """Run files across a small process pool.

    One file at a time is ~12x realtime -- LiteRT's own threading does not saturate the
    machine -- so the win comes from processing several recordings at once. But each worker
    holds a whole resampled recording in RAM, so the pool is sized by FREE MEMORY, not by
    cores, and a worker that still runs out is retried serially rather than failing the run.
    """
    import birdnet_lite as bl

    model_files = _model_files(model_dir())
    ts_regex = ts_format = None
    if recorder:
        prof = rec.get(recorder).resolve([p.name for p in paths])
        ts_regex, ts_format = prof.timestamp_spec()

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {str(p): (str(p), str(out_dir), lat, lon, week, min_conf, overlap, sensitivity,
                      ts_regex, ts_format) for p in paths}
    n = len(tasks)
    jobs = _plan_jobs(paths, jobs)
    if progress and jobs > 1:
        progress(0, n, f"{jobs} workers (~{_peak_gb_per_worker(paths):.1f} GB each)")

    written, done, deferred = [], 0, []
    if jobs > 1 and n > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                     initargs=model_files) as ex:
                futs = {ex.submit(_run_one, t): k for k, t in tasks.items()}
                for f in as_completed(futs):
                    try:
                        csv_path, name, k = f.result()
                    except MemoryError:
                        deferred.append(futs[f])
                        continue
                    done += 1
                    if progress:
                        progress(done, n, f"{name} ({k} detections)")
                    written.append(Path(csv_path))
        except MemoryError:
            deferred = [k for k in tasks if k not in {str(w) for w in written}]
    else:
        deferred = list(tasks)

    if deferred:                                 # serial retry: slower, but it finishes
        if progress and len(deferred) != n:
            progress(done, n, f"retrying {len(deferred)} file(s) serially (low memory)")
        _init_worker(*model_files)
        for key in deferred:
            csv_path, name, k = _run_one(tasks[key])
            done += 1
            if progress:
                progress(done, n, f"{name} ({k} detections)")
            written.append(Path(csv_path))
    return written


def _analyzer(audio_dir, out_dir, lat, lon, week, min_conf, overlap, sensitivity,
              threads, reprocess):
    cmd = [sys.executable, "-m", "birdnet_analyzer.analyze", str(audio_dir),
           "--output", str(out_dir), "--lat", str(lat), "--lon", str(lon),
           "--week", str(week if week is not None else -1), "--min_conf", str(min_conf),
           "--overlap", str(overlap), "--sensitivity", str(sensitivity),
           "--rtype", "csv", "--threads", str(threads)]
    if not reprocess:
        cmd.append("--skip_existing_results")
    subprocess.run(cmd, check=True)
    return sorted(Path(out_dir).glob("*.csv"))


def analyze(paths, out_dir, lat, lon, week=None, min_conf=0.25, recorder=None,
            overlap=0.0, sensitivity=1.0, threads=0, progress=None,
            engine="auto", audio_dir=None, reprocess=False, jobs=0):
    """Run inference over `paths`, writing result tables into `out_dir`.

    `week` is BirdNET's 1..48 location filter (None = derive per file, -1 = disabled).
    `jobs` sizes the litert process pool (0 = auto). Returns the result files written.
    """
    eng = resolve(engine)
    out_dir = Path(out_dir)
    paths = [Path(p) for p in paths]
    if eng == "litert":
        return _litert(paths, out_dir, lat, lon, week, min_conf, recorder,
                       overlap, sensitivity, progress, jobs)
    if audio_dir is None:
        raise ValueError("engine 'analyzer' needs audio_dir (it scans a folder, not a file list)")
    return _analyzer(audio_dir, out_dir, lat, lon, week, min_conf, overlap, sensitivity,
                     threads or max(2, os.cpu_count() or 4), reprocess)
