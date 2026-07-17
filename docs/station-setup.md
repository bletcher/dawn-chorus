# Station setup: capturing detections for `dawnchorus`

This is the capture half: a mic under a house eave, feeding detections to `dawnchorus`.
There are **two paths**, and the tool reads both:

- **Batch (recommended for research)** — a scheduled recorder (**AudioMoth** / **Song
  Meter**) captures only the dawn window; you run **BirdNET-Analyzer** over the files
  offline and point `dawnchorus --from-analyzer` at the result tables. **Keeps your raw
  audio** for verification/reprocessing; no always-on Pi. Jump to
  [Batch pipeline](#batch-pipeline-recommended).
- **Live station** — **BirdNET-Pi/Go** on an always-on Raspberry Pi identifies in real
  time and writes a SQLite DB that `dawnchorus --db` reads. Live dashboard; no raw-audio
  archive. See [Live station](#alternative-live-birdnet-pigo-station).

The hardware below (mic, windscreen, mounting, pass-through) is shared; only the compute
and the software differ.

---

## Parts list

Prices are rough 2026 USD and vary by supplier — treat product names as *classes of
thing*, not endorsements. Two builds: a **budget** one that works, and a **recommended**
one that's more robust for a season of unattended running.

### Essential

| Item | Suggested pick | ~USD | Notes |
|---|---|---|---|
| Computer | Raspberry Pi 5 (4 GB) — or Pi 4 (4 GB) for budget | 55–60 | 4 GB is ample; either runs BirdNET-Go fine |
| Power supply | Official USB-C PSU (27 W for Pi 5 / 15 W for Pi 4) | 10–15 | use the real one; undervolting causes weird faults |
| Cooling (Pi 5) | Official active cooler | 5 | Pi 5 wants it; Pi 4 can go passive |
| Case | Any vented case | 10–15 | indoors, so nothing fancy |
| Boot storage | USB SSD 240 GB (recommended) **or** SanDisk Max Endurance 64 GB microSD (budget) | 15–40 | 24/7 writes kill cheap SD cards; SSD boot is the durable choice |
| **Microphone A** *(better)* | Low-self-noise electret (EM272-class capsule) + USB sound card with plug-in power | 55–80 | see US sourcing note below; the sound card gives the capsule its bias voltage |
| **Microphone B** *(simplest)* | Decent USB omnidirectional mic / lavalier | 20–30 | one cable, no sound card; fine for species-level detection |
| Windscreen | Foam + furry "dead cat" (fitted if using EM272) | 8–15 | **essential outdoors** — wind roar wrecks detections |
| Mic housing | Small vented, downward-facing ABS/junction box | 10–15 | shelters the capsule, still lets it hear |
| Desiccant | Silica-gel packets | 8 | fights day/night condensation inside the housing |
| Mount + fixings | Bracket/clamp, screws, cable clips | 10–15 | to fix it rigidly under the eave |
| Cable pass-through | Rubber grommet + exterior silicone caulk | 8 | seal the hole; leave a drip loop |
| USB cable | Short **passive** USB extension (< 5 m) | 8 | keep the run short so passive USB works |

### Optional but smart

| Item | ~USD | Why |
|---|---|---|
| Small UPS / battery-backup plug | 40–70 | a storm flicker mid-write can corrupt storage; this prevents it |
| Gore/breather vent for the housing | 6 | equalizes humidity so the capsule doesn't sweat |
| Active USB extension (if run > 5 m) | 12 | passive USB tops out ~5 m |

### Rough totals

- **Budget build** (Pi 4 + Max-Endurance SD + USB omni mic): **~$150**
- **Recommended build** (Pi 5 + USB SSD + EM272 + housing + UPS): **~$260**

### Microphone sourcing (US, no European shipping)

Clippy/Micbooster ships from the UK. US-available alternatives, best-first:

- **AudioMoth as a USB mic** — flash the USB-microphone firmware and cable it to the Pi.
  Bioacoustics-standard, sold US-side via **GroupGets** (CA) and **LabMaker**; an IPX
  waterproof case is available (you'll add a cable gland for the USB exit). ~$90 + case.
  *If your lab already has AudioMoths, this is the zero-purchase option.*
- **EM272-class electret capsule from a US distributor** — a low-self-noise capsule such as
  **PUI Audio AOM-5024L-HD** (or a Primo EM272/EM172 capsule) from **DigiKey / Mouser**,
  wired to a 3.5 mm plug and a plug-in-power USB sound card. Same electrical idea as the
  Clippy, a little soldering, no overseas shipping. ~$15–30 + sound card.
- **Plug-and-play USB omni (Amazon US)** — a Movo/Maono/Fifine USB lavalier or mini omni.
  Noisier than the above but fine for species detection and truly one-cable. ~$20–30.

> **Different architecture, US-local:** Wildlife Acoustics (Maynard, MA) makes the **Song
> Meter Micro**, an all-in-one recorder. It logs audio to an SD card rather than streaming
> to a Pi, so you'd skip the station and instead batch-process its recordings through
> **BirdNET-Analyzer** offline into a database, then feed that to `dawnchorus`. More manual,
> but no Pi to maintain — worth it if you already own Song Meters.

## The one decision that matters most: two thresholds, not one

There are **two** confidence cutoffs in this pipeline, and they should not be the same:

1. **Capture threshold** (BirdNET-Go `threshold`) — what gets *written to the database*.
   Set this **low** (≈0.3–0.4). A detection you don't log is gone forever; disk is cheap.
2. **Analysis floor** (`dawnchorus --min-confidence`) — what you *count* when computing
   phenology. Set this **higher** (≈0.6–0.7). You can retighten this any time and re-run.

Capture permissively, filter analytically. The quantile-onset design in `dawnchorus` is
built to tolerate the occasional stray false positive, so erring toward *more* logged
detections helps you catch quiet, distant, early singers (which is exactly what onset is
about) without paying for it in biased timing.

---

## Batch pipeline (recommended)

Record the dawn window, analyze offline, keep the audio. Four steps:

**1. Schedule the recorder.** Configure the AudioMoth / Song Meter to record a window that
brackets civil dawn — e.g. **sunrise − 2.5 h to sunrise + 4 h** (both apps support
sunrise-relative schedules). Record **48 kHz, mono**. Power it from mains USB so you never
swap batteries. That ~6.5 h/day is ≈2 GB/day of WAV, so a 256 GB card holds a whole season.

> **Using a Song Meter (e.g. Micro 2)?** It's already weatherproof and self-contained, so
> skip most of the [parts list](#parts-list) — no Pi, mic, sound card, housing, or
> pass-through; just SD cards, power/batteries, a mount, and a windscreen. **Set its clock
> to UTC** and pass `--file-tz UTC` (cleanest — Song Meters don't auto-adjust for daylight
> saving, so a fixed local offset would shift summer onsets by an hour). Record **48 kHz
> mono** on a **sunrise-relative** schedule. `dawnchorus` reads its
> `PREFIX_YYYYMMDD_HHMMSS.wav` filenames out of the box.

**2. Offload.** Copy or swap the SD card on whatever cadence suits — weekly, monthly, or
once at season's end. **Keep the WAVs**; they're your primary data (compress to FLAC in the
archive to roughly halve the size).

**3. Analyze with BirdNET-Analyzer.** `pip install birdnet-analyzer`, then run it over the
folder writing one CSV per file. Use a **low** capture threshold and pass your location so
the range filter trims implausible species:

```bash
python -m birdnet_analyzer.analyze \
    --i /path/to/recordings --o /path/to/results \
    --lat 42.53 --lon -72.53 --week -1 \
    --min_conf 0.3 --rtype csv --threads 4
# (--week -1 = whole-year range filter; flags vary by version — check --help)
```

**4. Load into `dawnchorus`.** Point `--from-analyzer` at the results folder; it rebuilds
each detection's wall-clock time from the **recording filename + within-file offset**:

```bash
python -m dawnchorus.cli --from-analyzer /path/to/results \
    --lat 42.53 --lon -72.53 --tz America/New_York --file-tz UTC \
    --min-confidence 0.65 \
    --weather --weather-cache mystation_weather.csv --out results/
```

> **AudioMoth stamps filenames in UTC by default** — pass **`--file-tz UTC`** so times
> convert to your station tz. (If you instead set the recorder's clock to local time, omit
> it.) Get this wrong and every onset shifts by your whole UTC offset.

The two-threshold split applies here too: **BirdNET-Analyzer `--min_conf 0.3`** (capture
liberally) → **`dawnchorus --min-confidence 0.65`** (filter for analysis).

---

## Alternative: live BirdNET-Pi/Go station

If you'd rather have a live dashboard and don't need the raw-audio archive, run **BirdNET-Go**
on an always-on Pi; it identifies in real time and writes the SQLite DB that
`dawnchorus --db` reads. The mic hardware and the **Placement checklist** below are shared
with the batch path; the **First-run checklist** (flashing the Pi, setting the audio
device) is specific to this live station.

### BirdNET-Go config (a decision map)

> **Reconcile against the generated default.** BirdNET-Go writes a complete `config.yaml`
> on first run, and its field names shift between releases. Treat the block below as
> *"what to set and why,"* and apply each decision to the config your version generates
> rather than pasting this blind.

```yaml
birdnet:
  # Location drives the range filter (which species are plausible here) and is what
  # you'll echo to dawnchorus. Use your eave's real coordinates (4 decimals is plenty).
  latitude: 42.5300        # <-- REPLACE with your site
  longitude: -72.5300      # <-- REPLACE with your site
  locale: en

  sensitivity: 1.25        # 0.5–1.5; >1 = more sensitive. Up a notch to catch quiet /
                           # distant / pre-dawn singers. Costs some false positives.
  threshold: 0.4           # CAPTURE floor — log liberally, filter later. Keep it LOW.
  overlap: 1.5             # 0–2.9 s window overlap; ~1.5 gives finer onset timing at
                           # more CPU. Drop to 0 if the Pi can't keep up.
  threads: 0               # 0 = auto

realtime:
  interval: 0              # seconds to suppress repeat detections of the SAME species;
                           # 0 = keep them all (you want density for phenology)
  audio:
    source: "sysdefault"   # <-- set to your USB mic's ALSA device (see checklist)
    export:
      enabled: true        # keep clips so you can verify detections...
      type: flac           # ...compressed & lossless (~half the size of wav)
      path: clips/
      retention:
        policy: usage      # auto-prune so a season doesn't fill the disk
        maxusage: 80%      # start dropping oldest clips at 80% disk use
        minclips: 5        # but always keep a few examples per species

webserver:
  enabled: true
  port: 8080               # live dashboard at http://<pi-ip>:8080

output:
  sqlite:
    enabled: true          # <-- THE artifact dawnchorus reads
    path: birdnet.db
  mysql:
    enabled: false
```

### Pairing command (analysis side)

Once it's logged a few dawns, copy `birdnet.db` off and run:

```bash
python -m dawnchorus.cli --db birdnet.db --lat 42.53 --lon -72.53 \
    --tz America/New_York --min-confidence 0.65 \
    --weather --weather-cache mystation_weather.csv --out results/
```

Note the **0.4 capture → 0.65 analysis** split, and **always pass `--lat/--lon`** — some
BirdNET-Go builds store `0.0` in the row, which is unusable for solar/weather math.

---

## Placement checklist (mounting day)

- [ ] Back eave that faces the woods/fields; confirm **no HVAC condenser, heat-pump,
      dryer vent, or pump** on that wall.
- [ ] Mount ~3 m up, tucked under the overhang, capsule pointing **down-and-out** toward
      the habitat (sheds rain, cuts wall reflections and house noise).
- [ ] Fit the **foam + furry windscreen** (not optional outdoors).
- [ ] Capsule in a small **vented, downward-facing housing** with a **desiccant** packet.
- [ ] Run the USB cable (**< 5 m** for passive; else an *active* USB extension) through a
      grommeted hole; seal with exterior caulk; leave a **drip loop** outside.
- [ ] **Photograph and note the exact position.** It must not move all season —
      detectability is part of the signal.

## First-run checklist (setup day — live station)

- [ ] Flash **Raspberry Pi OS Lite**; enable SSH + WiFi in the imager; boot; `sudo apt
      update && sudo apt full-upgrade`.
- [ ] Install **BirdNET-Go** (Docker Compose *or* the project install script per its
      README); map a **persistent data volume/dir** so the DB and clips survive restarts.
- [ ] Plug in the USB mic; find its capture device: `arecord -l` → set
      `realtime.audio.source` (e.g. `plughw:1,0` or the card name).
- [ ] **Listen to a test clip:** `arecord -d 10 -f cd test.wav`, play it back. Check for
      usable level, **no clipping**, no constant hum/hiss. Adjust gain / mic angle.
- [ ] Set **latitude/longitude**, `threshold: 0.4`, `sensitivity: 1.25`, enable **SQLite
      output**, enable **clip export with retention**.
- [ ] Start the service; open `http://<pi-ip>:8080`; confirm detections appear with live
      spectrograms.
- [ ] Let it run one full dawn; next morning, confirm that morning's rows are in the DB.
- [ ] **End-to-end smoke test:** copy `birdnet.db` to your laptop, run the pairing command
      above, confirm CSVs + PNGs generate.
- [ ] Set up a recurring **DB pull** (scheduled `scp`/rsync over WiFi, or manual every few
      weeks) and skim clips occasionally to gauge the false-positive rate.

## Ongoing / seasonal

- [ ] **Don't move the mic.** Ever. (Worth repeating.)
- [ ] After storms, glance at the dashboard to confirm it's still logging.
- [ ] Watch disk usage the first two weeks to confirm clip retention is actually pruning.
- [ ] The chorus is a spring→summer story; run at least across those months before reading
      the seasonal-turnover tables.

---

## Getting your coordinates & timezone

- **Coordinates:** right-click your eave's spot in Google/Apple Maps, or stand there with
  a phone GPS. Solar math is insensitive to small errors — 4 decimals (~10 m) is overkill-
  accurate.
- **Timezone:** an IANA name, e.g. `America/New_York` for the US Northeast. This is what
  `dawnchorus --tz` needs; it's how clock time becomes correct solar time.
