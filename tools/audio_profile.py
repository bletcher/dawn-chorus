"""
Acoustic profile of two CO-LOCATED recorders over the same wall-clock minutes.

`compare_recorders.py` says *whether* two boxes disagree; `pair_detections.py` says the
disagreement is a confidence offset. Neither says WHY, and "why" is what decides which knob
to turn on the recorder. Since co-located boxes were exposed to identical air, any
difference in the waveform is the hardware plus its settings, and it is directly measurable:

  level        broadband RMS. A pure gain difference shifts this and nothing else.
  noise floor  10th percentile of per-frame energy - what the box hears when nothing is
               singing. This is the number that decides marginal detections: BirdNET scores
               signal-to-noise, so a raised floor costs confidence on distant, quiet birds
               while leaving loud close ones untouched. Exactly the pattern of a
               species-dependent confidence penalty.
  headroom     peak level and clipped-sample fraction. Clipping distorts loud calls and can
               *raise* detection counts for broadband species while wrecking tonal ones.
  band table   level and floor per frequency band, so a high-pass filter, a rolled-off mic,
               or band-limited noise (wind, self-noise, handling) shows up where it lives.

Read the per-band SNR proxy (loud percentile minus floor) as the actionable column: whichever
band loses SNR is the band whose species you are losing.

    python tools/audio_profile.py --a data --recorder-a song-meter-micro-2 \
        --b data_owl --recorder-b owl-sense --at "2026-08-07 05:00" --seconds 300
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd                                # noqa: E402
from dawnchorus import recorders as rec            # noqa: E402

# Bands chosen around what the dawn chorus actually occupies; 2-4 kHz is robin/thrush song.
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 6000), (6000, 8000), (8000, 12000)]


def find_slice(audio_dir, recorder, at: pd.Timestamp, seconds: float):
    """Locate the recording covering `at` and read `seconds` from that offset."""
    prof = rec.get(recorder)
    paths = sorted(p for p in Path(audio_dir).glob("*") if p.suffix.lower() == ".wav")
    conv = None
    if prof is not None:
        conv = rec.CONVENTIONS_BY_ID.get(prof.resolve([p.name for p in paths]).convention or "")
    conv = conv or rec.sniff([p.name for p in paths])
    for p in paths:
        t = conv.parse(p.name) if conv else None
        if t is None:
            continue
        with wave.open(str(p)) as w:
            sr, n = w.getframerate(), w.getnframes()
            dur = n / sr
            start = pd.Timestamp(t)
            off = (at - start).total_seconds()
            if 0 <= off <= dur - seconds:
                w.setpos(int(off * sr))
                raw = w.readframes(int(seconds * sr))
                sw = w.getsampwidth()
                if sw != 2:
                    raise SystemExit(f"{p.name}: {sw * 8}-bit not handled")
                x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
                if w.getnchannels() > 1:
                    x = x.reshape(-1, w.getnchannels()).mean(axis=1)
                return p.name, sr, x / 32768.0
    return None, None, None


def db(v):
    return 20.0 * np.log10(np.maximum(v, 1e-12))


def profile(x, sr, frame=4096):
    """Level / floor / headroom overall and per band, all in dBFS."""
    nfr = len(x) // frame
    X = x[:nfr * frame].reshape(nfr, frame)
    win = np.hanning(frame)
    spec = np.abs(np.fft.rfft(X * win, axis=1)) / (frame / 4)
    freqs = np.fft.rfftfreq(frame, 1 / sr)

    frame_rms = np.sqrt((X ** 2).mean(axis=1))
    out = {
        "rms_db": float(db(np.sqrt((x ** 2).mean()))),
        "floor_db": float(db(np.percentile(frame_rms, 10))),
        "loud_db": float(db(np.percentile(frame_rms, 95))),
        "peak_db": float(db(np.abs(x).max())),
        "clipped_%": float(100.0 * (np.abs(x) >= 0.999).mean()),
    }
    out["snr_db"] = out["loud_db"] - out["floor_db"]

    rows = []
    for lo, hi in BANDS:
        if lo >= sr / 2:
            continue
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        if not m.any():
            continue
        e = np.sqrt((spec[:, m] ** 2).sum(axis=1))       # per-frame energy in the band
        rows.append({"band": f"{lo // 1000 if lo >= 1000 else lo}{'k' if lo >= 1000 else ''}-"
                             f"{hi // 1000}k" if hi >= 1000 else f"{lo}-{hi}",
                     "level_db": float(db(np.percentile(e, 50))),
                     "floor_db": float(db(np.percentile(e, 10))),
                     "loud_db": float(db(np.percentile(e, 95)))})
    band = pd.DataFrame(rows)
    band["snr_db"] = band["loud_db"] - band["floor_db"]
    return out, band


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare two co-located recorders' audio")
    p.add_argument("--a", required=True); p.add_argument("--recorder-a", required=True)
    p.add_argument("--b", required=True); p.add_argument("--recorder-b", required=True)
    p.add_argument("--at", required=True, help='wall-clock start, e.g. "2026-08-07 05:00"')
    p.add_argument("--seconds", type=float, default=300.0)
    args = p.parse_args(argv)

    at = pd.Timestamp(args.at)
    got = {}
    for lbl, d, r in (("A", args.a, args.recorder_a), ("B", args.b, args.recorder_b)):
        name, sr, x = find_slice(d, r, at, args.seconds)
        if x is None:
            raise SystemExit(f"no recording in {d} covers {at} + {args.seconds}s")
        got[lbl] = (r, name, sr, x)
        print(f"{lbl} = {r:22} {name}  ({sr} Hz, {len(x) / sr:.0f}s from {at})")

    res = {k: profile(v[3], v[2]) for k, v in got.items()}
    ra, rb = got["A"][0], got["B"][0]

    print(f"\nOVERALL (dBFS)          {ra:>22} {rb:>22}   B-A")
    for k, lab in (("rms_db", "broadband RMS"), ("floor_db", "noise floor (p10)"),
                   ("loud_db", "loud frames (p95)"), ("snr_db", "SNR proxy (p95-p10)"),
                   ("peak_db", "peak"), ("clipped_%", "clipped samples %")):
        a, b = res["A"][0][k], res["B"][0][k]
        print(f"  {lab:22}{a:22.2f} {b:22.2f} {b - a:+7.2f}")

    ba, bb = res["A"][1], res["B"][1]
    m = ba.merge(bb, on="band", suffixes=("_a", "_b"))
    m["d_level"] = m["level_db_b"] - m["level_db_a"]
    m["d_floor"] = m["floor_db_b"] - m["floor_db_a"]
    m["d_snr"] = m["snr_db_b"] - m["snr_db_a"]
    print(f"\nPER BAND (B - A, dB).  d_snr < 0 means B has worse signal-to-noise there.")
    print(m[["band", "level_db_a", "level_db_b", "d_level",
             "floor_db_a", "floor_db_b", "d_floor", "d_snr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
