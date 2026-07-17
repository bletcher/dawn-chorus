# dawnchorus

Turn a BirdNET detection database into dawn-chorus phenology: **when** each species
starts singing relative to the sun, **how long** it stays vocally active, **how the
chorus composition turns over across the season**, and **how weather shifts the timing.**

It reads the SQLite database that BirdNET-Go or BirdNET-Pi already produces — no new
hardware, no forking your station — and emits tidy tables + reference plots.

## Getting the data: a BirdNET station

`dawnchorus` is the **analysis half**. It assumes something is already listening at your
site and logging timestamped detections to SQLite. That capture half is a solved
problem — you don't need to build it, and you don't need Merlin:

* **Why not Merlin?** Merlin Bird ID (Cornell) is a closed consumer app with **no public
  API** — there's no supported way to pull timestamped detections out of it for analysis.
  It's superb for identifying a bird in the moment, but it isn't a data source you can
  automate against.
* **Use BirdNET instead.** BirdNET (same lab: the K. Lisa Yang Center at Cornell) is the
  open acoustic model, and two turnkey projects run it unattended on cheap hardware and
  write **every detection to a SQLite database** — exactly what this tool consumes:
  * **[BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi)** / its maintained fork
    **[BirdNET-Go](https://github.com/tphakala/birdnet-go)** — a Raspberry Pi (3B+/4/5 or
    a Zero 2 W) + a USB or I²S microphone, listening 24/7. Web dashboard, live spectrograms,
    and the detection DB this tool reads. BirdNET-Go also runs as a Docker container on any
    always-on Linux box or mini-PC if you'd rather not use a Pi.

**Minimum recipe:** a Raspberry Pi, a weatherproof USB mic (e.g. an AudioMoth used as a
USB mic, a Clippy EM272, or any decent omnidirectional lavalier), power, and a roof/window
the same spot all season. Flash the image, let it log for a few dawns, then point
`dawnchorus` at the resulting `.db`. Keep the mic and its position **fixed** — detectability
is part of what you're measuring, so moving the mic mid-season breaks comparability.

> A phone running the Merlin app on the porch will *show* you the chorus, but it won't give
> you a queryable, timestamped record to compute onset/turnover from. If a phone is all you
> have, record continuous audio and run the detections through BirdNET-Analyzer offline into
> a database — but a dedicated always-on station is far less fuss.

## Why this exists

BirdNET stations answer *"what species are here?"* They don't answer *"when does the
Wood Thrush start, is it earlier in June than May, and is it earlier on warm
mornings?"* — because that needs every detection re-expressed in **solar time**
(minutes from civil dawn / sunrise for your lat-lon and date), reduced to
onset/offset/occupancy per species per morning, and optionally joined to weather.

## What it computes

**Per species, per morning** (`morning_summary.csv`): `onset_min`, `offset_min`,
`span_min`, `peak_min`, `occupancy`, `n_detections`, plus `raw_first_min`/`raw_last_min`.
With `--weather`, per-morning covariates are merged in.

**Per species, per month** (`species_phenology_month.csv`): median + IQR of onset and span.

**Seasonal turnover**: `composition_month.csv` (each species' share of the dawn window)
and `richness_month.csv` (mean species per morning + cumulative pool).

**Weather** (`--weather`): `morning_weather.csv` (per-morning conditions) and
`weather_response.csv` (per-species onset~covariate screen).

## Install & run

```bash
pip install -r requirements.txt     # astral, pandas, matplotlib, pyyaml, requests

# No hardware, no network — synthetic station near Montague, MA + mock weather cache:
python tools/make_synthetic_db.py demo.db demo_weather.csv
python -m dawnchorus.cli --db demo.db --lat 42.53 --lon -72.53 \
    --tz America/New_York --min-confidence 0.6 \
    --weather --weather-cache demo_weather.csv --out results/

# Your own station (fetches real weather from Open-Meteo):
python -m dawnchorus.cli --db /path/to/birds.db --lat <LAT> --lon <LON> \
    --tz America/New_York --config config.example.yaml \
    --weather --weather-cache mystation_weather.csv --out results/
```

Library use:

```python
import dawnchorus as dc
r = dc.run("demo.db", latitude=42.53, longitude=-72.53, tz="America/New_York",
           weather=True, weather_cache="demo_weather.csv")
r["morning_summary"]      # per species-morning, with weather columns
r["weather_response"]     # onset~temperature / ~cloud_cover slopes per species
```

## How onset is defined (the important choice)

The morning **window** ([dawn − 2 h, dawn + 4 h] by default) bounds which detections
count. Within it, **onset and offset are robust quantiles** of detection times — 5th
and 95th percentiles by default. Deliberately:

* No detection-count threshold, so it doesn't bias quiet species late (the failure
  mode of a "k detections in a rolling window" rule).
* A single stray daytime false positive barely moves a 5th percentile when real
  singing is present.
* Read it as *"the time by which 5% of the morning's detections have occurred,"* not
  the literal first note — a consistent, comparison-friendly definition. `raw_first_min`
  gives the literal earliest; set `onset_quantile: 0.0` for literal first/last.

Mornings with fewer than `min_detections_per_morning` (default 5) get NaN onset but
still count toward composition and richness.

## The ECDF: the robust full-distribution view

The scalar onset/offset are only two quantiles of a richer object: the **empirical
cumulative distribution** of a species' detection times. `species_ecdf()` builds it.
F(t) is the fraction of a species' vocal activity that has occurred by solar-minute t,
so onset = the 0.05 crossing, median song-time = 0.5, offset = 0.95 — all read off one
curve, and you compare whole singing schedules across months or weather strata instead
of comparing points. It is robust by construction: one spurious detection shifts an
ECDF by 1/n, a single negligible step.

The **morning is the replicate** (detections within a morning are pseudoreplicated), so
each morning's ECDF is computed on a common solar-time grid and then averaged across
mornings, with a p25–p75 band showing morning-to-morning spread. `pooled=True` gives
the simpler single-curve version.

* `species_ecdf.csv` — tidy grid table (period, species, t_min, F, F_p25, F_p75).
* `ecdf_quantiles.csv` — onset/median/offset (and any quantiles) read off each curve.
  These agree with `morning_summary` onset to a few minutes (mean-of-morning-curves vs
  median-of-morning-quantiles differ slightly; both robust).
* `ecdf_month_shifts.csv` — per species, consecutive-period **KS statistic**, **1-D
  Wasserstein distance in minutes** (how far the whole schedule moved), and median
  shift. These are pooled over detections, so treat them as effect-size *descriptors*,
  not tests — for inference, compare per-morning summaries with morning as the unit in
  R. `ecdf_by_month.png` overlays the monthly curves per species.

## Weather (Open-Meteo)

`--weather` pulls hourly weather for your location and date range from
[Open-Meteo](https://open-meteo.com)'s ERA5 archive — **no API key** — and reduces it
to one row per morning: temperature and cloud at the anchor, plus window means/sums of
temperature, cloud cover, precipitation, wind, and humidity. Joined onto
`morning_summary` and screened against onset in `weather_response.csv`.

* **Cache it.** `--weather-cache path.csv`; first run fetches, later runs read the file.
* **Archive lag.** ERA5 lags ~5 days; for very recent mornings use `--weather-source forecast`.
* **Licence.** Open-Meteo free tier is **non-commercial**, CC BY 4.0 (attribution),
  ~10k calls/day. For commercial use, self-host (AGPL) or a paid tier — relevant if this
  ever feeds a client deliverable rather than personal citizen science.

**`weather_response.csv` is a screen, not inference.** Per-species OLS slope + Pearson r
of onset vs each covariate — no p-values, no pooling. Negative onset~temperature =
earlier on warmer mornings. But for year-round residents, onset vs *absolute*
temperature is confounded with the seasonal temperature ramp and changing daylength. To
attribute a shift to weather, detrend first — regress onset on the temperature *anomaly*
(observed minus seasonal expectation) or include day-of-year — and treat species/day as
random effects. That belongs in R (lme4/glmmTMB); this tool gets you to the tidy table.

## Read this before trusting a number

* **BirdNET doesn't separate song from call** and works in 3-second windows, so
  `span_min` is a *vocal-activity* span (not a song-bout length) and `occupancy` is a
  proxy for the fraction of the morning a species was detectably vocalising.
* **Detectability ≠ singing.** Distance, wind, and mic directionality shape what's
  logged. Keep anchor and hardware fixed across the season.
* **Weather confounds season** for wide-season residents (see above).

## Schema support

The loader introspects the SQLite file and maps columns automatically:

* **BirdNET-Go** — `notes` (or newer `detections`): `scientific_name`, `common_name`,
  `confidence`, `date`, `time`, `latitude`, `longitude`.
* **BirdNET-Pi** — `detections`: `Sci_Name`, `Com_Name`, `Confidence`, `Date`, `Time`,
  `Lat`, `Lon`.

If lat/lon are stored as `0.0` (BirdNET-Go, location filtering off), pass `--lat/--lon`;
solar and weather both need a real location.

## Handing off to your own viz

The CSVs are tidy/long-form on purpose. For publication figures, pipe
`morning_summary.csv` or `species_phenology_month.csv` into Observable Framework or
ggplot2 (a ridgeline of onset-vs-dawn by month, or a stacked-area of
`composition_month`, is a few lines there).

## Development

```bash
pip install -e ".[dev]"     # editable install + pytest
pytest                       # runs the suite in tests/
```

The suite covers schema auto-detection (BirdNET-Go and BirdNET-Pi layouts), the
solar-time math and its polar day/night guard, quantile-onset robustness to false
positives, the ECDF primitives, and an **end-to-end regression test** that builds the
synthetic station and checks the pipeline recovers the planted temperature→onset signal
and species ordering.

## Layout

```text
dawnchorus/
  io.py          schema-detecting SQLite loader -> normalized frame
  solar.py       civil-dawn / sunrise anchoring (astral); polar day/night guard
  phenology.py   quantile onset / offset / span / occupancy per species-morning
  seasonal.py    composition drift + richness
  ecdf.py        empirical CDF of vocal activity: curves, quantiles, KS/Wasserstein
  weather.py     Open-Meteo fetch, per-morning summary, onset~weather screen
  plots.py       reference matplotlib figures
  cli.py         end-to-end runner
tools/
  make_synthetic_db.py   demo DB + mock weather cache (temp-driven onset)
tests/           pytest suite (unit + end-to-end signal-recovery)
```

MIT. BirdNET model and station software are separate projects by their authors
(K. Lisa Yang Center / Cornell; tphakala; Nachtzuster; P. McGuire). Weather data ©
Open-Meteo contributors, CC BY 4.0.
