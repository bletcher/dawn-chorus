"""
Minimal BirdNET inference on LiteRT — no TensorFlow, so it bundles small (for the desktop app).

BirdNET's .tflite is end-to-end: raw 48 kHz audio in (3 s = 144000 samples), 6522 species logits
out (the mel-spectrogram + CNN are inside the model). So inference is just: load audio, split into
3 s chunks, run the interpreter, sigmoid the logits. This mirrors birdnet_analyzer's own pipeline
(audio.split_signal + model.flat_sigmoid) exactly, but without importing TensorFlow.

    interp, labels = load(model_path, labels_path)
    dets = analyze_file("REC_20260725_063502.wav", interp, labels, min_conf=0.25)
    # dets: list of {datetime, scientific_name, common_name, confidence}

Deps: ai-edge-litert, librosa, numpy. Recordings never leave the machine.
"""
from __future__ import annotations

import re
from datetime import timedelta

import librosa
import numpy as np
from ai_edge_litert.interpreter import Interpreter

SAMPLE_RATE = 48000
SIG_LENGTH = 3.0
TS_REGEX = r"\d{8}_\d{6}"
TS_FORMAT = "%Y%m%d_%H%M%S"


def load(model_path: str, labels_path: str):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    labels = [ln.strip() for ln in open(labels_path, encoding="utf-8") if ln.strip()]
    return interp, labels


def split_signal(sig, rate=SAMPLE_RATE, seconds=SIG_LENGTH, overlap=0.0, minlen=1.0):
    """Port of birdnet_analyzer.audio.split_signal (USE_NOISE=False): non-overlapping (or
    overlapping) chunks of `seconds`, last chunk zero-padded, dropped if under `minlen`."""
    chunksize = int(rate * seconds)
    stepsize = int(rate * (seconds - overlap))
    minsize = int(rate * minlen)
    lastchunkpos = int((sig.size - chunksize + stepsize - 1) / stepsize) * stepsize
    if lastchunkpos < 0:
        lastchunkpos = 0
    elif sig.size - lastchunkpos < minsize:
        lastchunkpos -= stepsize
    data = np.concatenate((sig, np.zeros(chunksize, dtype=sig.dtype)))
    return [data[i:i + chunksize] for i in range(0, lastchunkpos + 1, stepsize)]


def _flat_sigmoid(logits, sensitivity=1.0, bias=1.0):
    """birdnet_analyzer.model.flat_sigmoid — CLI --sensitivity maps to -sensitivity here."""
    transformed_bias = (bias - 1.0) * 10.0
    return 1.0 / (1.0 + np.exp(-sensitivity * np.clip(logits + transformed_bias, -20, 20)))


def _file_start(name, ts_regex=None, ts_format=None):
    """Recording start parsed from the filename.

    Defaults to the AudioMoth / Song Meter `YYYYMMDD_HHMMSS` convention. Callers that know
    the recorder (desktop/app.py resolves a `dawnchorus.recorders` profile) pass that
    profile's convention instead, so this module stays free of dawnchorus imports and can
    keep being bundled on its own.
    """
    from datetime import datetime, timezone
    ts_regex = ts_regex or TS_REGEX
    ts_format = ts_format or TS_FORMAT
    m = re.search(ts_regex, str(name))
    if not m:
        return None
    if ts_format == "hexepoch":                     # legacy AudioMoth: hex UTC unix epoch
        try:
            return datetime.fromtimestamp(int(m.group(0), 16), tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError, OverflowError):
            return None
    try:
        return datetime.strptime(m.group(0), ts_format)
    except ValueError:
        return None


def analyze_file(path, interp, labels, min_conf=0.25, overlap=0.0, sensitivity=1.0, allowed=None,
                 ts_regex=None, ts_format=None):
    """Run BirdNET on one recording; return detections with reconstructed wall-clock times.

    `allowed` (optional bool mask over the 6522 species from `species_mask`) applies BirdNET's
    location/week filter, dropping species implausible for the site.
    `ts_regex`/`ts_format` override the filename timestamp convention for other recorders."""
    sig, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, res_type="kaiser_fast")
    chunks = split_signal(sig, SAMPLE_RATE, SIG_LENGTH, overlap, 1.0)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    start = _file_start(path, ts_regex, ts_format)
    step = SIG_LENGTH - overlap
    dets = []
    for i, chunk in enumerate(chunks):
        interp.set_tensor(inp["index"], np.asarray(chunk, dtype=np.float32).reshape(1, 144000))
        interp.invoke()
        conf = _flat_sigmoid(interp.get_tensor(out["index"])[0], sensitivity)
        for j in np.where(conf >= min_conf)[0]:
            if allowed is not None and not allowed[j]:
                continue
            sci, _, com = labels[j].partition("_")
            offset = i * step
            dets.append({"datetime": (start + timedelta(seconds=offset)) if start else None,
                         "offset_s": round(float(offset), 1),
                         "scientific_name": sci, "common_name": com or sci,
                         "confidence": round(float(conf[j]), 4)})
    return dets


# --- location/week filter (BirdNET's MData model) --------------------------------------------
LOCATION_FILTER_THRESHOLD = 0.03


def load_meta(mdata_path):
    m = Interpreter(model_path=mdata_path)
    m.allocate_tensors()
    return m


def week_from_date(d):
    """BirdNET's 48-week convention (4 weeks/month); -1 for yearlong."""
    return (d.month - 1) * 4 + min((d.day - 1) // 7 + 1, 4)


def species_mask(meta, lat, lon, week, threshold=LOCATION_FILTER_THRESHOLD):
    """Bool mask over the 6522 species that are plausible at (lat, lon, week)."""
    inp = meta.get_input_details()[0]
    out = meta.get_output_details()[0]
    meta.set_tensor(inp["index"], np.array([[lat, lon, week]], dtype=np.float32))
    meta.invoke()
    return meta.get_tensor(out["index"])[0] >= threshold


def analyze_folder(folder, interp, labels, lat, lon, meta=None, min_conf=0.25,
                   overlap=0.0, sensitivity=1.0, progress=None, ts_regex=None, ts_format=None):
    """Analyze every .wav in a folder; per-file location filter from its date. Returns a list of
    detection dicts. `progress(done, total, name)` is called per file if given.
    `ts_regex`/`ts_format` override the filename timestamp convention for other recorders."""
    import glob
    import os
    wavs = sorted(set(glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.WAV"))))
    rows = []
    for k, wav in enumerate(wavs):
        allowed = None
        if meta is not None:
            fs = _file_start(os.path.basename(wav), ts_regex, ts_format)
            allowed = species_mask(meta, lat, lon, week_from_date(fs) if fs else -1)
        rows.extend(analyze_file(wav, interp, labels, min_conf, overlap, sensitivity, allowed,
                                 ts_regex, ts_format))
        if progress:
            progress(k + 1, len(wavs), os.path.basename(wav))
    return rows
