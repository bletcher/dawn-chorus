"""
Event-level pairing of two CO-LOCATED recorders: did they hear the same individual calls?

ONLY VALID FOR CO-LOCATED BOXES. If the two recorders sat side by side, they were exposed
to the same acoustic scene, so every difference between them is hardware — not distance,
not aspect, not a different bird. That is a far stronger design than comparing aggregates
from two positions, and it lets us ask a question aggregates cannot answer:

    when box A logged a call and box B did not, did B *mishear* it or *not hear* it at all?

`compare_recorders.py` counts detections per species. That conflates two very different
failures. This matches individual detections in time (same species, within `--tol` seconds)
and splits A's detections three ways:

    both        B also cleared the analysis floor  -> the boxes agree
    B scored low B has a detection there, but only below the floor -> a GAIN/SENSITIVITY
                offset. B heard the bird and rated it lower; a threshold change would
                recover it, and the bias is systematic and correctable.
    B silent    B has nothing there even at the capture floor -> B genuinely missed it
                (or A produced a false positive).

The confidence distribution of each bucket is the tell. If A's unmatched detections sit
near the floor, the boxes differ only in how they score faint calls. If they are confident
detections that B never registered, the boxes are not interchangeable regardless of what
the onset medians say.

    python tools/pair_detections.py \
        --recorder-a song-meter-micro-2 --results-a data/results \
        --recorder-b owl-sense          --results-b data_owl/results \
        --lat 42.537278 --lon -72.531694 --tz America/New_York --species "American Robin"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dawnchorus as dc                                        # noqa: E402
from compare_recorders import (audio_coverage, overlap_windows,  # noqa: E402
                               restrict_to_windows)
from dawnchorus import recorders as rec                        # noqa: E402


def pair_one_species(a: pd.DataFrame, b: pd.DataFrame, floor: float, tol_s: float) -> pd.DataFrame:
    """For every A detection >= floor, find B's nearest detection of the same species.

    Returns A's rows with `conf_b` (NaN when B has nothing within tol) and a `bucket`.
    """
    a = a[a["confidence"] >= floor].sort_values("datetime")
    b = b.sort_values("datetime")[["datetime", "confidence"]].rename(
        columns={"confidence": "conf_b"})
    if a.empty:
        return pd.DataFrame(columns=["datetime", "confidence", "conf_b", "bucket"])
    if b.empty:
        out = a[["datetime", "confidence"]].copy()
        out["conf_b"] = float("nan")
        out["bucket"] = "B silent"
        return out
    m = pd.merge_asof(a[["datetime", "confidence"]], b, on="datetime",
                      tolerance=pd.Timedelta(seconds=tol_s), direction="nearest")
    m["bucket"] = pd.cut(m["conf_b"].fillna(-1), [-2, -0.5, floor, 1.01],
                         labels=["B silent", "B scored low", "both"], right=False)
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Event-level pairing of two co-located recorders")
    p.add_argument("--recorder-a", required=True)
    p.add_argument("--results-a", required=True)
    p.add_argument("--audio-a", default=None)
    p.add_argument("--recorder-b", required=True)
    p.add_argument("--results-b", required=True)
    p.add_argument("--audio-b", default=None)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    # Pinned at 0.50 on purpose: a diagnostic, not a chart. The recorder bake-off
    # was concluded at 0.50, and its numbers should stay reproducible.
    p.add_argument("--min-confidence", type=float, default=0.5, help="analysis floor")
    p.add_argument("--tol", type=float, default=5.0,
                   help="seconds within which two detections are the same call (default 5; "
                        "BirdNET's 3 s windows sit on different grids when the files start "
                        "at different seconds)")
    p.add_argument("--species", default=None, help="limit to one common name")
    p.add_argument("--top", type=int, default=12, help="how many species to table")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    la, lb = args.recorder_a, args.recorder_b
    det = {}
    for lbl, res, recorder in ((la, args.results_a, args.recorder_a),
                               (lb, args.results_b, args.recorder_b)):
        det[lbl] = dc.load_birdnet_analyzer(res, min_confidence=0.0, latitude=args.lat,
                                            longitude=args.lon, tz=args.tz, recorder=recorder)
        det[lbl]["datetime"] = pd.to_datetime(det[lbl]["datetime"])

    # Same mornings, same minutes - otherwise "B silent" just means "B wasn't recording".
    if args.audio_a and args.audio_b:
        windows = overlap_windows(audio_coverage(args.audio_a, rec.get(args.recorder_a)),
                                  audio_coverage(args.audio_b, rec.get(args.recorder_b)))
        det = {k: restrict_to_windows(v, windows) for k, v in det.items()}
        print(f"restricted to mutual coverage on {len(windows)} morning(s): "
              + ", ".join(f"{d} {lo:%H:%M}-{hi:%H:%M}" for d, (lo, hi) in sorted(windows.items())))
    else:
        print("! no --audio-a/--audio-b: 'B silent' may just mean B was not recording.")

    floor = args.min_confidence
    names = ([args.species] if args.species else
             det[la][det[la]["confidence"] >= floor]["common_name"]
             .value_counts().head(args.top).index.tolist())

    rows, detail = [], {}
    for sp in names:
        a = det[la][det[la]["common_name"] == sp]
        b = det[lb][det[lb]["common_name"] == sp]
        m = pair_one_species(a, b, floor, args.tol)
        detail[sp] = m
        c = m["bucket"].value_counts()
        n = len(m)
        rows.append({"species": sp, "A_dets": n,
                     "both": int(c.get("both", 0)),
                     "B_low": int(c.get("B scored low", 0)),
                     "B_silent": int(c.get("B silent", 0)),
                     "agree_%": round(100 * c.get("both", 0) / n, 1) if n else float("nan"),
                     "medconf_B_low": round(float(m.loc[m["bucket"] == "B scored low",
                                                        "confidence"].median()), 3),
                     "medconf_B_silent": round(float(m.loc[m["bucket"] == "B silent",
                                                           "confidence"].median()), 3)})
    tab = pd.DataFrame(rows)
    print(f"\nA = {la}   B = {lb}   floor {floor}   tol +/-{args.tol:g}s")
    print("(medconf_* = A's confidence on the calls B scored low / missed entirely)\n")
    print(tab.to_string(index=False, na_rep="-"))

    tot = tab[["A_dets", "both", "B_low", "B_silent"]].sum()
    if tot["A_dets"]:
        print(f"\noverall: {tot['both']}/{tot['A_dets']} agree "
              f"({100 * tot['both'] / tot['A_dets']:.0f}%), "
              f"{tot['B_low']} scored low by B, {tot['B_silent']} missed by B entirely")
        print("  'scored low' is a correctable gain/threshold offset; 'missed entirely' is not.")

    if args.out:
        outp = Path(args.out)
        outp.mkdir(parents=True, exist_ok=True)
        tab.to_csv(outp / "pairing_summary.csv", index=False)
        pd.concat([d.assign(species=s) for s, d in detail.items()]).to_csv(
            outp / "pairing_detail.csv", index=False)
        print(f"\nwrote {outp}/pairing_summary.csv + pairing_detail.csv")


if __name__ == "__main__":
    main()
