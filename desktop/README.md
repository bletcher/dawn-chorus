# Dawn Chorus desktop app

A one-click app so a contributor can turn a folder of recordings into a publishable site — **no
Python install, no TensorFlow**. Recordings never leave their machine; only detection times are
written.

```
pick recordings folder  +  site name + coordinates  →  Run  →  <slug>.json  →  publish
```

## Why this exists (and why it's small)

Running BirdNET in the *browser* is blocked: the model's mel front-end uses an `RFFT2D` op that
the browser WASM TFLite runtime can't execute (it loads, then aborts). Full TensorFlow would
bundle to ~1 GB. The unlock is **LiteRT** (`ai-edge-litert`) — a lightweight TFLite runtime that
runs BirdNET's `.tflite` (RFFT2D and all) with no TensorFlow, so the app bundles to a fraction of
that.

`birdnet_lite.py` is the inference engine. It's validated **byte-for-byte** against
BirdNET-Analyzer on real data: **480/480 detections reproduced, max confidence Δ 0.0002**, with
BirdNET's identical location/week species filter. End-to-end it reproduces the hosted
`montague.json` exactly (37 species / 1421 detections). It is not an approximation.

## Run from source (development)

```bash
cd desktop
pip install -r requirements.txt
pip install -e ..                    # the dawnchorus analysis engine
# models: set BIRDNET_MODELS to a birdnet-analyzer checkpoints/V2.4 folder, or stage them (see build.ps1)
python app.py                        # GUI
python app.py --cli --folder <wavs> --slug mysite --name "My Site" \
    --lat 42.5 --lon -72.5 --tz America/New_York --out out/
```

## Build the one-click executable

`build.ps1` stages the three models into `models/` and runs PyInstaller:

```powershell
./build.ps1                          # -> desktop/dist/dawn-chorus/dawn-chorus.exe
```

It bundles LiteRT + librosa + the models (~few hundred MB). The build is Windows here; run the
same on macOS/Linux to produce those targets. `models/`, `build/`, `dist/` are git-ignored.

## Files

- `birdnet_lite.py` — TF-free BirdNET inference on LiteRT (load → split → invoke → sigmoid →
  location filter). Reusable on its own.
- `app.py` — the GUI + `--cli` pipeline: folder → inference → `build_payload` → `<slug>.json`.
- `build.ps1` — stage models + PyInstaller build.

## Publishing the result

The app writes `<slug>.json` (+ updates `sites.json`). To put a site live: drop those into the
web repo's `site/data/` and push (or open a PR / send the file to the maintainer). Same payload
contract as the rest of the platform, so it just appears in the picker.
