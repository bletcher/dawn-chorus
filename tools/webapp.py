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
import subprocess
import sys
import threading
import time
from itertools import count
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException                       # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse         # noqa: E402
from fastapi.staticfiles import StaticFiles                      # noqa: E402
from pydantic import BaseModel                                   # noqa: E402

import config                                                    # noqa: E402
import track                                                     # noqa: E402

app = FastAPI(title="dawn-chorus local")

_jobs: dict[int, dict] = {}
_ids = count(1)
_lock = threading.Lock()


class RunIn(BaseModel):
    command: str                      # process | dashboard | compare | publish | all
    site: str | None = None
    deployment: str | None = None
    push: bool = False


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
    """Deployments plus what's unprocessed - the numbers the buttons act on."""
    out = []
    for d in config.deployments():
        rec = {"site": d.site, "key": d.key, "label": d.label, "name": d.name,
               "recorder": d.recorder, "unit": d.unit, "audio": d.audio.name,
               "note": d.note, "exists": d.audio.exists(),
               "recordings": 0, "processed": 0, "new": 0, "dashboard": None}
        if d.audio.exists():
            audio = track.scan(str(d.audio))
            man = track.load_manifest(str(d.manifest))
            new, changed, _ = track.diff(audio, man)
            rec.update(recordings=len(audio), processed=len(man.get("files", {})),
                       new=len(new) + len(changed))
        page = ROOT / "site" / f"dashboard-{d.key}.html"
        if page.exists():
            rec["dashboard"] = f"/site/{page.name}"
        out.append(rec)
    return {"deployments": out}


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


def _mount_static():
    """Serve the dashboards and the recordings from this same origin.

    StaticFiles honours HTTP Range, which click-to-listen needs: without it every click
    would pull a whole 165-330 MB recording instead of seeking into it.
    """
    site = ROOT / "site"
    if site.exists():
        app.mount("/site", StaticFiles(directory=str(site)), name="site")
    for d in config.deployments():
        if d.audio.exists():
            app.mount(f"/{d.audio.name}", StaticFiles(directory=str(d.audio)), name=d.audio.name)


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
