"""
Read `deployments.json`: which sites exist and which recorder listens at each.

One source of truth for the CLI wrapper and the local web app, so a site's coordinates,
timezone and recorder id are declared once. That matters more than convenience: a wrong
`--recorder` silently mis-parses filenames (dropping whole recordings) or assumes the wrong
clock zone (shifting every solar time by the UTC offset), and neither failure raises.

    from config import load, deployment
    cfg  = load()
    d    = deployment("montague", "owl")     # -> Deployment(audio=..., recorder=..., site=...)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawnchorus import recorders as rec              # noqa: E402

CONFIG_PATH = ROOT / "deployments.json"


@dataclass(frozen=True)
class Deployment:
    site: str            # slug
    key: str             # deployment id within the site, e.g. "sm"
    name: str            # site display name
    lat: float
    lon: float
    tz: str
    audio: Path          # folder of recordings for THIS recorder
    recorder: str        # profile id in dawnchorus.recorders
    unit: str | None
    note: str = ""

    @property
    def results(self) -> Path:
        return self.audio / "results"

    @property
    def manifest(self) -> Path:
        return self.audio / "manifest.json"

    @property
    def label(self) -> str:
        return f"{self.site}/{self.key}"


def _audio_path(raw: str) -> Path:
    """Resolve a deployment's audio folder.

    Absolute paths are used as given, so recordings can live on a big external drive
    (`F:/dev/dawn-chorus/data`) while the repo stays on the system disk -- a full card
    dump is several GB per session. Relative paths stay relative to the repo root, which
    keeps the default layout working unchanged.

    Everything for a deployment travels together inside this folder: the recordings, the
    `results/` tables and `manifest.json`. Pointing `audio` somewhere new without moving
    those makes the deployment look unprocessed and re-runs the whole card.
    """
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. It lists your sites and the recorder at each; "
            "copy the one in the repo root and edit it for your location.")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not cfg.get("sites"):
        raise ValueError(f"{p} has no 'sites'")
    return cfg


def deployments(site: str | None = None, key: str | None = None,
                path: str | Path | None = None) -> list[Deployment]:
    """All deployments, optionally narrowed to one site and/or one deployment key."""
    cfg = load(path)
    out = []
    for slug, s in cfg["sites"].items():
        if site and slug != site:
            continue
        for k, d in s.get("deployments", {}).items():
            if key and k != key:
                continue
            # Fail loudly here rather than let a typo become a silent mis-parse downstream.
            rec.get(d["recorder"])
            out.append(Deployment(
                site=slug, key=k, name=s["name"], lat=float(s["lat"]), lon=float(s["lon"]),
                tz=s["tz"], audio=_audio_path(d["audio"]), recorder=d["recorder"],
                unit=d.get("unit"), note=d.get("note", "")))
    if not out:
        raise KeyError(f"no deployment matches site={site!r} key={key!r}. "
                       f"Known: {', '.join(d.label for d in deployments(path=path))}"
                       if site or key else "no deployments configured")
    return out


def deployment(site: str, key: str, path: str | Path | None = None) -> Deployment:
    return deployments(site, key, path)[0]


def primary(site: str, path: str | Path | None = None) -> Deployment:
    """The site's reference recorder - the one whose numbers form its published series."""
    cfg = load(path)
    s = cfg["sites"][site]
    k = s.get("primary") or next(iter(s["deployments"]))
    return deployment(site, k, path)


def describe(path: str | Path | None = None) -> str:
    lines = []
    cfg = load(path)
    for slug, s in cfg["sites"].items():
        prim = s.get("primary")
        lines.append(f"{slug}  {s['name']}  ({s['lat']}, {s['lon']}, {s['tz']})")
        for d in deployments(slug, path=path):
            star = " *primary" if d.key == prim else ""
            n = len(list(d.audio.glob("*.wav"))) if d.audio.exists() else 0
            lines.append(f"    {d.key:<6} {d.recorder:<20} {str(d.audio.name):<10} "
                         f"{n:>4} recordings{star}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())


def exclusions(site: str, path: str | Path | None = None) -> list[dict]:
    """Curated non-bird exclusions for a site: [{date, species[], reason}, ...]."""
    cfg = load(path)
    return cfg.get("sites", {}).get(site, {}).get("exclusions", []) or []


def apply_exclusions(det, site: str, path=None):
    """Drop excluded (date, species) rows. Returns (filtered, notes).

    Never silent: the caller prints the notes and the payload carries them, so a reader
    can always see what was removed and why.
    """
    rules = exclusions(site, path)
    if not rules or det is None or not len(det):
        return det, []
    import pandas as pd
    dates = pd.to_datetime(det["datetime"]).dt.date.astype(str)
    notes, keep = [], pd.Series(True, index=det.index)
    for r in rules:
        m = (dates == str(r["date"])) & det["common_name"].isin(r["species"])
        n = int(m.sum())
        if n:
            keep &= ~m
            notes.append({"date": r["date"], "species": list(r["species"]),
                          "removed": n, "reason": r.get("reason", "")})
    return det[keep].copy(), notes
