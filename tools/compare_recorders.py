"""
Head-to-head comparison of two recorders running at the same place on the same mornings.

Two boxes on one post do not produce the same numbers. Mic sensitivity, self-noise, gain
and sample rate all move BirdNET's confidence scores, and `morning_summary` derives onset
from a quantile of *detections above a confidence floor* - so hardware differences land
directly on the project's headline metric. This tool measures that gap instead of assuming
it away, and answers the questions a bake-off actually turns on:

  1. Do the two clocks agree?      (checked FIRST - every later number is garbage if not)
  2. Did they cover the same mornings, for the same length of time?
  3. Does one hear more birds - more detections, more species, higher confidence?
  4. **Do they agree on onset?**   The one that matters: if swapping boxes shifts onset by
     more than the day-to-day biological signal, the two are not interchangeable and the
     site's time series cannot span the swap without a correction.

    python tools/compare_recorders.py \
        --recorder-a song-meter-micro-2 --results-a data/results \
        --recorder-b owl-sense          --results-b data_owl/results \
        --lat 42.537278 --lon -72.531694 --tz America/New_York \
        --audio-a data --audio-b data_owl --out compare/

`--audio-*` is optional; given, it reads WAV headers to report true recorded minutes and
sample rate (so detection counts can be read per recorded hour, not per card).
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dawnchorus as dc                                   # noqa: E402
from dawnchorus import recorders as rec                   # noqa: E402
from dawnchorus.phenology import DEFAULTS, _anchor_col    # noqa: E402

MINUTES_PER_DAY = 1440


# --------------------------------------------------------------------------------------
# 1. Clock agreement
# --------------------------------------------------------------------------------------

def clock_lag(a: pd.DataFrame, b: pd.DataFrame, windows: dict | None = None) -> dict:
    """Minutes recorder B's clock runs ahead of A's, from their daily activity curves.

    Both boxes hear the same chorus, so their detection density over the time-of-day
    should peak together. We pool detections into a minute-of-day histogram and circularly
    cross-correlate. A clean peak near 0 means the clocks agree; a peak near a whole UTC
    offset (+/-240 min here) means one is stamping UTC - the most invisible failure here.

    CRITICAL: correlate only over the window where BOTH boxes were recording. Detection
    density is dominated by whether a recorder was switched on at all, so two boxes on
    different schedules produce histograms that are essentially their duty cycles; the
    correlation then measures schedule overlap and reports a confident, entirely fictional
    lag. Restricting to mutual coverage leaves biology as the only thing that can align.

    Returns lag_min plus `sharpness` (peak / median score). A flat correlation (sharpness
    near 1) means too few detections to conclude anything, so we say so rather than
    reporting a confident zero.
    """
    if windows:
        a, b = restrict_to_windows(a, windows), restrict_to_windows(b, windows)
    if len(a) < 20 or len(b) < 20:
        return {"lag_min": None, "sharpness": 0.0, "confident": False, "n": min(len(a), len(b))}
    def hist(df):
        t = pd.to_datetime(df["datetime"])
        mod = (t.dt.hour * 60 + t.dt.minute).to_numpy()
        h = np.bincount(mod, minlength=MINUTES_PER_DAY).astype(float)
        return h - h.mean()          # centre, so the correlation isn't dominated by volume

    ha, hb = hist(a), hist(b)
    if ha.std() == 0 or hb.std() == 0:
        return {"lag_min": None, "sharpness": 0.0, "confident": False, "n": min(len(a), len(b))}

    lags = np.arange(-MINUTES_PER_DAY // 2, MINUTES_PER_DAY // 2)
    scores = np.array([float(np.dot(ha, np.roll(hb, -int(L)))) for L in lags])
    best = int(np.argmax(scores))
    peak, typical = scores[best], np.median(np.abs(scores))
    sharpness = float(peak / typical) if typical > 0 else 0.0
    return {"lag_min": int(lags[best]), "sharpness": sharpness,
            "confident": sharpness >= 3.0, "n": min(len(a), len(b))}


def interpret_lag(lag: dict, tz_offset_hours: float | None) -> list[str]:
    """Turn the measured lag into the specific thing the operator should do about it."""
    if lag["lag_min"] is None or not lag["confident"]:
        return ["  ! too few detections to establish clock agreement - treat every number "
                "below as provisional."]
    L = lag["lag_min"]
    if abs(L) <= 5:
        return [f"  OK clocks agree (lag {L:+d} min). Recorder B needs no tz conversion."]
    msgs = [f"  ! recorder B's clock runs {L:+d} min relative to A."]
    if tz_offset_hours is not None and abs(abs(L) - abs(tz_offset_hours) * 60) <= 10:
        msgs.append(f"    That is the station's UTC offset ({tz_offset_hours:+g} h): B is almost "
                    "certainly stamping filenames in UTC.")
        msgs.append("    Fix: set clock=\"utc\" on B's profile in dawnchorus/recorders.py "
                    "(or pass --file-tz-b UTC), then re-run this comparison.")
    elif abs(L) % 60 <= 3 or abs(L) % 60 >= 57:
        msgs.append("    A whole number of hours - a time-zone or DST setting, not drift.")
    else:
        msgs.append("    Not a whole hour - looks like real clock drift or a mis-set time. "
                    "Check B's clock against a known reference before trusting its onsets.")
    return msgs


# --------------------------------------------------------------------------------------
# 2. Coverage
# --------------------------------------------------------------------------------------

COVERAGE_COLS = ["date", "minutes", "sample_rate", "channels", "start", "end"]


def audio_coverage(audio_dir, profile=None) -> pd.DataFrame:
    """Per-morning recorded minutes, sample rate, and the wall-clock span actually covered.

    `start`/`end` matter as much as the totals: two boxes on different schedules only
    overlap for part of the morning, and a species whose onset falls before the later box
    powered on would be scored as "late" on that box. That is a coverage artefact, not a
    hardware difference, so the caller intersects these spans before comparing onsets.
    """
    conv = None
    if profile is not None and profile.convention:
        conv = rec.CONVENTIONS_BY_ID.get(profile.convention)
    rows = []
    # One glob, filtered by suffix: globbing "*.wav" AND "*.WAV" double-counts every file
    # on Windows, where the pattern match is already case-insensitive - which silently
    # halves every det/recorded-hour rate.
    for p in sorted(q for q in Path(audio_dir).glob("*") if q.suffix.lower() == ".wav"):
        c = conv or rec.sniff([p.name])
        t = c.parse(p.name) if c else None
        if t is None:
            continue
        try:
            with wave.open(str(p)) as w:
                dur = w.getnframes() / w.getframerate()
                rows.append({"date": t.date(), "minutes": dur / 60.0,
                             "sample_rate": w.getframerate(), "channels": w.getnchannels(),
                             "start": pd.Timestamp(t),
                             "end": pd.Timestamp(t) + pd.Timedelta(seconds=dur)})
        except Exception:
            continue                                   # unreadable/partial file; skip it
    if not rows:
        return pd.DataFrame(columns=COVERAGE_COLS)
    df = pd.DataFrame(rows)
    return (df.groupby("date")
              .agg(minutes=("minutes", "sum"), sample_rate=("sample_rate", "max"),
                   channels=("channels", "max"), start=("start", "min"), end=("end", "max"))
              .reset_index())


def overlap_windows(cov_a: pd.DataFrame, cov_b: pd.DataFrame) -> dict:
    """{date: (start, end)} where BOTH recorders were actually running."""
    out = {}
    if cov_a.empty or cov_b.empty:
        return out
    a = cov_a.set_index("date")
    b = cov_b.set_index("date")
    for d in a.index.intersection(b.index):
        lo = max(a.loc[d, "start"], b.loc[d, "start"])
        hi = min(a.loc[d, "end"], b.loc[d, "end"])
        if hi > lo:
            out[d] = (lo, hi)
    return out


def restrict_to_windows(det: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Keep only detections inside each morning's mutual-coverage window."""
    if not windows:
        return det
    t = pd.to_datetime(det["datetime"])
    mask = pd.Series(False, index=det.index)
    for d, (lo, hi) in windows.items():
        mask |= (det["date"] == d) & (t >= lo) & (t < hi)
    return det[mask].copy()


# --------------------------------------------------------------------------------------
# 3/4. Species and onset agreement
# --------------------------------------------------------------------------------------

def species_table(det_a, det_b, label_a, label_b) -> pd.DataFrame:
    """Per-species detection counts and median confidence for each recorder."""
    def side(d):
        return d.groupby("common_name").agg(n=("confidence", "size"),
                                            conf=("confidence", "median"))
    sa, sb = side(det_a), side(det_b)
    out = sa.join(sb, how="outer", lsuffix="_a", rsuffix="_b").fillna(
        {"n_a": 0, "n_b": 0})
    out = out.rename(columns={"n_a": f"n_{label_a}", "n_b": f"n_{label_b}",
                              "conf_a": f"conf_{label_a}", "conf_b": f"conf_{label_b}"})
    out["detected_by"] = np.where(
        (out[f"n_{label_a}"] > 0) & (out[f"n_{label_b}"] > 0), "both",
        np.where(out[f"n_{label_a}"] > 0, label_a, label_b))
    return out.reset_index().sort_values(
        [f"n_{label_a}", f"n_{label_b}"], ascending=False).reset_index(drop=True)


def onset_agreement(ms_a, ms_b, label_a, label_b) -> pd.DataFrame:
    """Paired per (morning, species) onset/offset/peak, with B-minus-A deltas.

    Restricted to pairs where BOTH recorders produced a defined onset, so the deltas
    measure disagreement about timing rather than one box simply missing a species.
    """
    keys = ["date", "scientific_name", "common_name"]
    cols = keys + ["n_detections", "onset_min", "offset_min", "peak_min"]
    m = ms_a[cols].merge(ms_b[cols], on=keys, suffixes=(f"_{label_a}", f"_{label_b}"))
    for f in ("onset_min", "offset_min", "peak_min"):
        m[f"d_{f}"] = m[f"{f}_{label_b}"] - m[f"{f}_{label_a}"]
    return m.dropna(subset=["d_onset_min"]).reset_index(drop=True)


def _fmt_stats(x: pd.Series) -> str:
    if x.empty:
        return "n/a"
    return (f"median {x.median():+.1f}  IQR [{x.quantile(.25):+.1f}, {x.quantile(.75):+.1f}]  "
            f"|median| {x.abs().median():.1f}  n={len(x)}")


# --------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Compare two recorders deployed at one site")
    p.add_argument("--recorder-a", required=True, help=f"profile id ({', '.join(sorted(rec.REGISTRY))})")
    p.add_argument("--results-a", required=True, help="BirdNET-Analyzer result folder for A")
    p.add_argument("--audio-a", default=None, help="A's WAV folder (optional: recorded minutes, sample rate)")
    p.add_argument("--recorder-b", required=True)
    p.add_argument("--results-b", required=True)
    p.add_argument("--audio-b", default=None)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--file-tz-a", dest="file_tz_a", default=None,
                   help="override A's clock zone (else the profile decides)")
    p.add_argument("--file-tz-b", dest="file_tz_b", default=None)
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="analysis floor, matching the dashboard (default 0.5)")
    p.add_argument("--out", default=None, help="folder for the CSVs (optional)")
    args = p.parse_args(argv)

    la, lb = args.recorder_a, args.recorder_b
    print("=" * 78)
    print(f"RECORDER COMPARISON  -  {la}  vs  {lb}")
    print("=" * 78)

    # Load each side under its own profile: the profile supplies the filename convention
    # and clock zone, and tags each row so the two never silently pool.
    det = {}
    for lbl, results, recorder, ftz in ((la, args.results_a, args.recorder_a, args.file_tz_a),
                                        (lb, args.results_b, args.recorder_b, args.file_tz_b)):
        names = [f.name for f in Path(results).rglob("*") if f.is_file()]
        print(f"\n[{lbl}] {rec.describe(recorder, names)}")
        prof = rec.get(recorder)
        if prof is not None and prof.needs_clock_check and ftz is None:
            print("        clock zone is UNCONFIRMED for this profile - the check below settles it.")
        d = dc.load_birdnet_analyzer(results, min_confidence=0.0, latitude=args.lat,
                                     longitude=args.lon, tz=args.tz, file_tz=ftz,
                                     recorder=recorder)
        det[lbl] = d
        print(f"        {len(d):,} raw detections, {d['date'].nunique()} morning(s), "
              f"{d['common_name'].nunique()} species (before the {args.min_confidence} floor)")

    # Coverage is needed BEFORE the clock check: the correlation is only meaningful over
    # the minutes both boxes were actually recording (see clock_lag).
    days = {lbl: set(d["date"].unique()) for lbl, d in det.items()}
    shared = sorted(days[la] & days[lb])
    cov = {}
    for lbl, adir, recorder in ((la, args.audio_a, args.recorder_a),
                                (lb, args.audio_b, args.recorder_b)):
        if adir:
            cov[lbl] = audio_coverage(adir, rec.get(recorder))
    windows = overlap_windows(cov.get(la, pd.DataFrame(columns=COVERAGE_COLS)),
                              cov.get(lb, pd.DataFrame(columns=COVERAGE_COLS)))

    # ---- 1. clocks -------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. CLOCK AGREEMENT")
    print("-" * 78)
    lag = clock_lag(det[la], det[lb], windows)
    if windows:
        print("  (restricted to the minutes both boxes were recording)")
    else:
        print("  ! no --audio-a/--audio-b: correlating over each box's FULL schedule. If they "
              "ran\n    different hours this measures schedule overlap, not clock offset.")
    off = None
    try:
        import zoneinfo
        sample = pd.to_datetime(det[la]["datetime"]).iloc[0]
        off = sample.tz_localize(zoneinfo.ZoneInfo(args.tz)).utcoffset().total_seconds() / 3600
    except Exception:
        pass
    print(f"  cross-correlation of daily activity: lag {lag['lag_min']} min "
          f"(peak sharpness {lag['sharpness']:.1f}x)")
    for line in interpret_lag(lag, off):
        print(line)

    # ---- 2. coverage -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. COVERAGE")
    print("-" * 78)

    def fmt(dates):
        return ", ".join(str(d) for d in sorted(dates)) or "none"

    print(f"  {la:22} mornings: {fmt(days[la])}")
    print(f"  {lb:22} mornings: {fmt(days[lb])}")
    print(f"  shared mornings       : "
          f"{fmt(shared) if shared else 'NONE - nothing below is comparable'}")
    for lbl in (la, lb):
        c = cov.get(lbl)
        if c is not None and not c.empty:
            # Show EVERY rate present. A folder whose settings changed mid-series looks
            # uniform if you print only the max, and that is exactly the case worth seeing.
            rates = sorted(set(int(r) for r in c["sample_rate"].dropna()))
            desc = ", ".join(f"{r:,} Hz ({r / 2000:.1f} kHz Nyquist)" for r in rates)
            if len(rates) > 1:
                desc += "  <- MIXED: this folder's settings changed mid-series"
            print(f"  {lbl:22} {c['minutes'].sum():.0f} recorded min, {desc}, "
                  f"{int(c['channels'].max())} ch")
            for r in c.itertuples():
                print(f"      {str(r.date)}  {r.start:%H:%M}-{r.end:%H:%M}")
    if len(cov) == 2 and all(not c.empty for c in cov.values()):
        # Rate over the SHARED mornings only. Taking the max across the whole folder
        # reports a setting that was never in play during the comparison -- this archive
        # switched from 24 to 48 kHz on 2026-08-08, long after the paired mornings.
        def _rate(lbl):
            c = cov[lbl]
            c = c[c["date"].isin(shared)] if shared else c
            return int(c["sample_rate"].max()) if not c.empty else None
        ra, rb = _rate(la), _rate(lb)
        if ra and rb and ra != rb:
            print(f"  ! sample rates differ ({ra:,} vs {rb:,} Hz). The lower box cannot represent "
                  f"anything above {min(ra, rb) / 2000:.1f} kHz, so some high-frequency species")
            print("    are unavailable to it by physics, not by hearing. Expect species-list gaps.")
    if not shared:
        print("\nNo overlapping mornings - stopping. Re-run once both cards cover the same dates.")
        return

    # Everything from here compares like with like: shared mornings, same floor.
    keep = {lbl: d[(d["confidence"] >= args.min_confidence) & (d["date"].isin(shared))].copy()
            for lbl, d in det.items()}

    # ...and, when we know each box's true recording span, the same CLOCK MINUTES too.
    # Without this, whichever recorder powered on later looks systematically "late".
    if windows:
        before = {lbl: len(d) for lbl, d in keep.items()}
        keep = {lbl: restrict_to_windows(d, windows) for lbl, d in keep.items()}
        print("\n  mutual-coverage window per morning (both boxes actually running):")
        for d in sorted(windows):
            lo, hi = windows[d]
            print(f"      {str(d)}  {lo:%H:%M}-{hi:%H:%M}  ({(hi - lo).total_seconds() / 60:.0f} min)")
        dropped = {lbl: before[lbl] - len(keep[lbl]) for lbl in keep}
        if any(dropped.values()):
            print("      restricted to it: dropped "
                  + ", ".join(f"{n} from {lbl}" for lbl, n in dropped.items())
                  + " outside mutual coverage")
    elif args.audio_a and args.audio_b:
        print("\n  ! the two recorders' spans never overlap on a shared morning - "
              "onset differences below would be schedule, not hardware.")
    else:
        print("\n  ! no --audio-a/--audio-b, so mutual coverage is unknown. If the two boxes ran "
              "different schedules, whichever started later will look systematically late.")

    # ---- 3. volume + species ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"3. WHAT EACH BOX HEARD  (shared mornings, confidence >= {args.min_confidence})")
    print("-" * 78)
    # Denominator: the mutual window if we restricted to one, else each box's own minutes.
    win_min = sum((hi - lo).total_seconds() / 60 for lo, hi in windows.values()) if windows else 0
    for lbl in (la, lb):
        d = keep[lbl]
        per = d.groupby("date").size()
        extra = ""
        mins = win_min
        if not mins and lbl in cov and not cov[lbl].empty:
            mins = cov[lbl][cov[lbl]["date"].isin(shared)]["minutes"].sum()
        if mins > 0:
            extra = f", {len(d) / (mins / 60):.0f} det/recorded-hour"
        print(f"  {lbl:22} {len(d):,} detections, {d['common_name'].nunique()} species"
              f"  ({', '.join(f'{k}: {v}' for k, v in per.items())}){extra}")

    sp = species_table(keep[la], keep[lb], la, lb)
    both = sp[sp["detected_by"] == "both"]
    only_a = sp[sp["detected_by"] == la]
    only_b = sp[sp["detected_by"] == lb]
    print(f"\n  species both: {len(both)}   only {la}: {len(only_a)}   only {lb}: {len(only_b)}")
    if not only_a.empty:
        print(f"  only {la}: {', '.join(only_a['common_name'].head(12))}")
    if not only_b.empty:
        print(f"  only {lb}: {', '.join(only_b['common_name'].head(12))}")
    if not both.empty:
        dconf = (both[f"conf_{lb}"] - both[f"conf_{la}"]).median()
        ratio = both[f"n_{lb}"].sum() / max(both[f"n_{la}"].sum(), 1)
        print(f"  on shared species: {lb} logs {ratio:.2f}x {la}'s detections; "
              f"median confidence {dconf:+.3f}")

    # ---- 4. onset ---------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. ONSET AGREEMENT  (the metric the project exists to measure)")
    print("-" * 78)
    solar = dc.SolarModel(args.lat, args.lon, args.tz)
    ms = {lbl: dc.morning_summary(solar.annotate(d)) for lbl, d in keep.items()}
    pair = onset_agreement(ms[la], ms[lb], la, lb)
    if pair.empty:
        print("  no (morning, species) pair had a defined onset on both recorders - "
              "too few detections. Nothing to conclude about interchangeability.")
    else:
        print(f"  paired (morning, species) onsets: {len(pair)}  "
              f"across {pair['common_name'].nunique()} species")
        for f, name in (("d_onset_min", "onset"), ("d_offset_min", "offset"), ("d_peak_min", "peak")):
            print(f"    {name:7} {lb} - {la}:  {_fmt_stats(pair[f].dropna())}  min")
        worst = pair.reindex(pair["d_onset_min"].abs().sort_values(ascending=False).index)
        print(f"\n  largest onset disagreements:")
        for r in worst.head(8).itertuples():
            print(f"    {str(r.date):11} {r.common_name:26} {r.d_onset_min:+7.1f} min")
        med = pair["d_onset_min"].abs().median()
        print(f"\n  VERDICT: typical onset disagreement is {med:.0f} min.")
        print("    Compare that against the day-to-day biological variation you are trying to")
        print("    detect. If it is the same size or larger, these two boxes are NOT")
        print("    interchangeable and a site's series must not span a swap uncorrected.")
        if windows:
            print("    NB: both sides were clipped to the mutual-coverage window, which makes the")
            print("    DIFFERENCE fair but leaves the absolute onsets truncated - don't read these")
            print("    onset values as the site's real phenology.")

    # ---- CSVs -------------------------------------------------------------------------
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sp.to_csv(out / "species_compare.csv", index=False)
        pair.to_csv(out / "onset_compare.csv", index=False)
        for lbl in (la, lb):
            ms[lbl].to_csv(out / f"morning_summary_{lbl}.csv", index=False)
            if lbl in cov and not cov[lbl].empty:
                cov[lbl].to_csv(out / f"coverage_{lbl}.csv", index=False)
        print(f"\nwrote CSVs to {out}/")


if __name__ == "__main__":
    main()
