"""
Local control panel: pick a deployment, click Run, watch it work, see the updated dashboard.

WHY THIS IS LOCAL AND NOT PART OF THE PUBLIC SITE
  Inference cannot happen in the browser (BirdNET's .tflite has an internal RFFT2D mel op
  that loads in tfjs-tflite WASM but aborts at inference), and the audio is far too big to
  upload - a couple of mornings is several GB. So the model has to run on this machine.
  A page served from https://<the public site> also cannot call http://127.0.0.1 at all;
  browsers block it as mixed content. Hence: this app serves the UI *and* the audio *and*
  the API from one local origin, and the public site stays a static publish target.

    python tools/webapp.py            ->  http://127.0.0.1:8765

Jobs are run as subprocesses of tools/run.py rather than in-process: inference uses a
process pool of its own, a crash then can't take the server down, and stdout is already
the progress format we want to stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from itertools import count
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException, Request              # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse         # noqa: E402
from fastapi.staticfiles import StaticFiles                      # noqa: E402
from pydantic import BaseModel                                   # noqa: E402
from starlette.routing import Mount                              # noqa: E402

import config                                                    # noqa: E402
import track                                                     # noqa: E402

app = FastAPI(title="dawn-chorus local")

_jobs: dict[int, dict] = {}
_ids = count(1)
_lock = threading.Lock()

# Duration is the only expensive field in a file listing (a header read per recording), and
# it cannot change for a given path+size. Cache it so polling for new arrivals stays cheap
# even with a card's worth of files on a spinning external drive.
_DUR: dict[tuple[str, int], float | None] = {}


def _duration_min(path: Path, size: int) -> float | None:
    key = (str(path), size)
    if key not in _DUR:
        try:
            with wave.open(str(path)) as w:
                _DUR[key] = round(w.getnframes() / w.getframerate() / 60.0, 1)
        except Exception:
            _DUR[key] = None
    return _DUR[key]


def _signature(audio: dict) -> str:
    """Cheap fingerprint of a folder's contents, for detecting arrivals without a full scan."""
    h = hashlib.sha1()
    for name in sorted(audio):
        h.update(f"{name}:{audio[name]['size']}\n".encode())
    return h.hexdigest()[:16]


class RunIn(BaseModel):
    command: str                      # process | dashboard | compare | publish | all
    site: str | None = None
    deployment: str | None = None
    push: bool = False
    only: list[str] | None = None     # run exactly these recordings


def _safe_name(name: str) -> str:
    """Reject anything that isn't a plain .wav filename.

    Uploads are written straight into a configured audio folder, so a name containing a
    path separator or `..` would let a request write outside it.
    """
    clean = Path(name).name
    if clean != name or not clean or clean.startswith("."):
        raise HTTPException(400, f"bad filename {name!r}")
    if not clean.lower().endswith(".wav"):
        raise HTTPException(400, "only .wav recordings are accepted")
    return clean


def _file_rows(d) -> list[dict]:
    """Every recording in a deployment, with whether it has been analysed.

    The `starts` field is parsed with THIS deployment's recorder convention, so a file the
    pipeline could not place in time shows up here as `unparseable` instead of being
    silently dropped during a run -- the failure mode that cost us every Owl Sense file.
    """
    from dawnchorus import recorders as rec

    audio = track.scan(str(d.audio))
    manifest = track.load_manifest(str(d.manifest))
    # NB: diff() takes the WHOLE manifest and unwraps "files" itself. Handing it the inner
    # dict makes every recording look unprocessed.
    new, changed, gone = track.diff(audio, manifest)
    man = manifest.get("files", {})
    new_s, changed_s = set(new), set(changed)

    prof = rec.get(d.recorder)
    conv = None
    if prof is not None:
        conv = rec.CONVENTIONS_BY_ID.get(prof.resolve(list(audio)).convention or "")
    conv = conv or rec.sniff(list(audio))

    rows = []
    for name in sorted(audio):
        info = man.get(name, {})
        start = conv.parse(name) if conv else None
        dur = _duration_min(d.audio / name, audio[name]["size"])
        if start is None:
            status = "unparseable"
        elif name in changed_s:
            status = "changed"
        elif name in new_s:
            status = "new"
        else:
            status = "processed"
        rows.append({
            "name": name, "status": status,
            "starts": start.strftime("%Y-%m-%d %H:%M:%S") if start else None,
            "minutes": dur, "mb": round(audio[name]["size"] / 1e6, 1),
            "detections": info.get("detections"),
            "processed_at": (info.get("processed_at") or "")[:16] or None,
            "recorder": info.get("recorder"),
        })
    for name in sorted(set(gone)):
        rows.append({"name": name, "status": "missing", "starts": None, "minutes": None,
                     "mb": None, "detections": man.get(name, {}).get("detections"),
                     "processed_at": None, "recorder": None})
    return rows


def _spawn(cmd_args: list[str]) -> int:
    """Start tools/run.py in the background, collecting its output for the UI."""
    jid = next(_ids)
    job = {"id": jid, "argv": cmd_args, "lines": [], "status": "running",
           "started": time.time(), "finished": None, "returncode": None}
    with _lock:
        _jobs[jid] = job

    def worker():
        # -u so progress arrives line by line instead of in a buffered lump at the end.
        proc = subprocess.Popen([sys.executable, "-u", str(HERE / "run.py"), *cmd_args],
                                cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                with _lock:
                    job["lines"].append(line)
                    del job["lines"][:-400]          # cap: this is a live log, not an archive
        proc.wait()
        with _lock:
            job["returncode"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "failed"
            job["finished"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return jid


@app.get("/api/state")
def state():
    """Deployments plus what's unprocessed - the numbers the buttons act on.

    Deliberately cheap (a stat per file, no audio headers) so the page can poll it while
    open and notice recordings that appeared underneath it -- copied from a card in
    Explorer, say, rather than uploaded through the panel. `sig` fingerprints the folder
    so the client can tell "something arrived" from "nothing changed" without re-reading
    the whole listing.
    """
    out = []
    for d in config.deployments():
        rec = {"site": d.site, "key": d.key, "label": d.label, "name": d.name,
               "recorder": d.recorder, "unit": d.unit, "audio": str(d.audio),
               "folder": d.audio.name, "note": d.note, "exists": d.audio.exists(),
               "recordings": 0, "processed": 0, "new": 0, "sig": "", "dashboard": None}
        if d.audio.exists():
            audio = track.scan(str(d.audio))
            man = track.load_manifest(str(d.manifest))
            new, changed, _ = track.diff(audio, man)
            rec.update(recordings=len(audio), processed=len(man.get("files", {})),
                       new=len(new) + len(changed), sig=_signature(audio))
        page = ROOT / "site" / f"dashboard-{d.key}.html"
        if page.exists():
            rec["dashboard"] = f"/site/{page.name}"
        out.append(rec)
    return {"deployments": out, "storage": _storage()}


def _storage() -> dict:
    """Where recordings live, and whether that drive can take another card."""
    paths = [d.audio for d in config.deployments()]
    parents = {p.parent for p in paths}
    root = str(parents.pop()) if len(parents) == 1 else None
    info = {"root": root, "shared": root is not None,
            "free_gb": None, "total_gb": None, "writable": False}
    probe = Path(root) if root else (paths[0] if paths else None)
    if probe:
        try:
            u = shutil.disk_usage(str(probe if probe.exists() else probe.anchor))
            info["free_gb"] = round(u.free / 1e9, 1)
            info["total_gb"] = round(u.total / 1e9, 1)
        except Exception:
            pass
        info["writable"] = os.access(str(probe), os.W_OK) if probe.exists() else False
    return info


class StorageIn(BaseModel):
    root: str


@app.get("/api/storage")
def storage():
    return _storage()


@app.post("/api/browse")
def browse():
    """Open a native folder picker on this machine and return the chosen path.

    Run in a subprocess: a Tk dialog created on a server worker thread is unreliable on
    Windows, and a wedged dialog would take the whole panel down with it.
    """
    code = ("import tkinter as tk;from tkinter import filedialog;"
            "r=tk.Tk();r.withdraw();r.attributes('-topmost',True);"
            "print(filedialog.askdirectory(title='Choose the recordings folder') or '')")
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=180)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "the folder picker was left open") from None
    chosen = (p.stdout or "").strip().splitlines()
    return {"path": chosen[-1] if chosen and chosen[-1] else None}


@app.post("/api/storage")
def set_storage(s: StorageIn):
    """Repoint every deployment at <root>/<its current folder name>.

    This MOVES NOTHING. A deployment's recordings, results/ and manifest.json travel
    together, so pointing at a location that doesn't already hold them makes the
    deployment look unprocessed and a run would re-analyse the whole card. We therefore
    report, per deployment, what is actually present at the new path and let the caller
    decide -- rather than silently starting a multi-hour re-run.
    """
    root = Path(s.root.strip().strip('"')).expanduser()
    if not root.is_absolute():
        raise HTTPException(400, "give an absolute path, e.g. F:/dev/dawn-chorus")
    if not root.exists():
        raise HTTPException(400, f"{root} does not exist")
    if not root.is_dir():
        raise HTTPException(400, f"{root} is not a folder")
    if not os.access(str(root), os.W_OK):
        raise HTTPException(400, f"{root} is not writable")

    cfg = json.loads(config.CONFIG_PATH.read_text(encoding="utf-8"))
    moved, notes = [], []
    for slug, site in cfg["sites"].items():
        for key, dep in site.get("deployments", {}).items():
            old = config._audio_path(dep["audio"])
            new = root / old.name
            dep["audio"] = new.as_posix()
            n_wav = len(list(new.glob("*.wav"))) if new.exists() else 0
            has_man = (new / "manifest.json").exists()
            moved.append({"deployment": f"{slug}/{key}", "from": str(old), "to": str(new),
                          "recordings": n_wav, "manifest": has_man})
            if not new.exists():
                notes.append(f"{slug}/{key}: {new} does not exist yet - it will be created "
                             "on the next upload, but any existing recordings are still at "
                             f"{old}.")
            elif n_wav == 0:
                notes.append(f"{slug}/{key}: {new} holds no recordings.")
            elif not has_man:
                notes.append(f"{slug}/{key}: {new} has recordings but no manifest.json, so "
                             "all of them will count as unprocessed.")
    config.CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    _mount_static(remount=True)
    return {"root": str(root), "deployments": moved, "warnings": notes}


@app.get("/api/files")
def files(site: str, deployment: str):
    d = config.deployment(site, deployment)
    if not d.audio.exists():
        raise HTTPException(404, f"no folder {d.audio}")
    return {"deployment": d.label, "recorder": d.recorder, "audio": str(d.audio),
            "files": _file_rows(d)}


@app.put("/api/files/{site}/{deployment}/{name}")
async def upload(site: str, deployment: str, name: str, request: Request):
    """Stream one recording into the deployment's folder.

    Raw PUT rather than multipart: these are 165-330 MB files, and streaming the body
    straight to disk avoids buffering a whole recording in memory (and avoids needing
    python-multipart). The upload is written to a .part file and renamed only on success,
    so an interrupted transfer can never look like a complete recording waiting to be run.
    """
    d = config.deployment(site, deployment)
    if not d.audio.exists():
        raise HTTPException(404, f"no folder {d.audio}")
    clean = _safe_name(name)
    dest = d.audio / clean
    if dest.exists():
        raise HTTPException(409, f"{clean} already exists")

    tmp = d.audio / f"{clean}.part"
    total = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                total += len(chunk)
        if total == 0:
            raise HTTPException(400, "empty upload")
        tmp.replace(dest)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"upload failed: {e}") from e

    # Tell the caller straight away whether this name can actually be placed in time.
    from dawnchorus import recorders as rec
    prof = rec.get(d.recorder)
    conv = rec.CONVENTIONS_BY_ID.get(prof.resolve([clean]).convention or "") if prof else None
    start = conv.parse(clean) if conv else None
    return {"name": clean, "bytes": total,
            "starts": start.strftime("%Y-%m-%d %H:%M:%S") if start else None,
            "warning": None if start else
                       (f"{clean} carries no timestamp this recorder's convention "
                        f"({d.recorder}) can read, so it cannot be placed in time and will "
                        "be skipped by a run.")}


@app.post("/api/run")
def run(r: RunIn):
    if r.command not in {"process", "dashboard", "compare", "publish", "all"}:
        raise HTTPException(400, f"unknown command {r.command!r}")
    argv = [r.command]
    if r.site:
        argv += ["--site", r.site]
    if r.deployment and r.command in {"process", "dashboard", "all"}:
        argv += ["--deployment", r.deployment]
    if r.push and r.command == "publish":
        argv += ["--push"]
    for n in (r.only or []):
        argv += ["--only", _safe_name(n)]
    return {"job": _spawn(argv)}


@app.get("/api/jobs/{jid}")
def job(jid: int, since: int = 0):
    with _lock:
        j = _jobs.get(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return JSONResponse({"id": j["id"], "status": j["status"], "returncode": j["returncode"],
                             "elapsed": round((j["finished"] or time.time()) - j["started"], 1),
                             "lines": j["lines"][since:], "n": len(j["lines"])})


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "webapp.html").read_text(encoding="utf-8")


def _mount_static(remount: bool = False):
    """Serve the dashboards and the recordings from this same origin.

    StaticFiles honours HTTP Range, which click-to-listen needs: without it every click
    would pull a whole 165-330 MB recording instead of seeking into it.

    Mount names are the FOLDER names, so a dashboard's embedded `../data` URL keeps
    working after the recordings move to another drive -- only the directory behind the
    mount changes. `remount` rebuilds them after the storage root is repointed, so the
    audio does not 404 until the server is restarted.
    """
    if remount:
        keep = {"site"}
        app.router.routes = [r for r in app.router.routes
                             if not (isinstance(r, Mount) and r.name not in keep
                                     and r.path != "/site")]
    site = ROOT / "site"
    if not any(isinstance(r, Mount) and r.path == "/site" for r in app.router.routes):
        if site.exists():
            app.mount("/site", StaticFiles(directory=str(site)), name="site")
    for d in config.deployments():
        if d.audio.exists():
            path = f"/{d.audio.name}"
            if not any(isinstance(r, Mount) and r.path == path for r in app.router.routes):
                app.mount(path, StaticFiles(directory=str(d.audio)), name=d.audio.name)


def main(argv=None):
    p = argparse.ArgumentParser(description="Dawn-chorus local control panel")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="keep this loopback-only: the app runs commands on this machine")
    args = p.parse_args(argv)

    _mount_static()
    import uvicorn
    print(f"\n  Dawn Chorus control panel -> http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
