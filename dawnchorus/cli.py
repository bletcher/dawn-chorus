"""
Command-line runner.

    python -m dawnchorus.cli --db demo.db --lat 42.53 --lon -72.53 \
        --tz America/New_York --weather --out results/

Writes tidy CSVs (morning_summary, species_phenology, composition, richness, and
with --weather: morning_weather, weather_response) plus reference PNGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from . import (load_detections, load_birdnet_analyzer, SolarModel, morning_summary,
               species_phenology, fetch_hourly, morning_weather, attach_weather,
               weather_response)
from .seasonal import composition, richness
from .ecdf import species_ecdf, ecdf_quantiles, ecdf_distance
from .phenology import DEFAULTS
from . import plots


def _load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        import yaml
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="Dawn-chorus phenology from BirdNET detections")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="BirdNET-Pi/Go SQLite database (live-station path)")
    src.add_argument("--from-analyzer", dest="from_analyzer",
                     help="folder/file of BirdNET-Analyzer result tables (batch path)")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", required=True, help="IANA tz, e.g. America/New_York")
    p.add_argument("--file-tz", dest="file_tz", default=None,
                   help="tz the recording FILENAMES are stamped in (e.g. UTC for AudioMoth); "
                        "converted to --tz. Omit if filenames are already station-local.")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--config", default=None)
    p.add_argument("--weather", action="store_true", help="fetch + join Open-Meteo covariates")
    p.add_argument("--weather-cache", default=None, help="CSV cache path for hourly weather")
    p.add_argument("--weather-source", default="archive", choices=["archive", "forecast"])
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.db:
        det = load_detections(args.db, min_confidence=args.min_confidence,
                              latitude=args.lat, longitude=args.lon)
    else:
        det = load_birdnet_analyzer(args.from_analyzer, min_confidence=args.min_confidence,
                                    latitude=args.lat, longitude=args.lon,
                                    tz=args.tz, file_tz=args.file_tz)
    solar = SolarModel(args.lat, args.lon, args.tz)
    det = solar.annotate(det)
    print(f"loaded {len(det):,} detections, {det['scientific_name'].nunique()} species, "
          f"{det['date'].nunique()} days")

    ms = morning_summary(det, cfg)

    if args.weather:
        dates = sorted(det["date"].unique())
        hourly = fetch_hourly(args.lat, args.lon, min(dates), max(dates), args.tz,
                              cache_path=args.weather_cache, source=args.weather_source)
        wx = morning_weather(hourly, solar, dates, config=cfg)
        ms = attach_weather(ms, wx)
        wx.to_csv(out / "morning_weather.csv", index=False)
        wr = weather_response(ms)
        wr.to_csv(out / "weather_response.csv", index=False)
        print(f"weather joined; onset~covariate screen -> {len(wr)} species-covariate fits")

    sp_month = species_phenology(ms, by="month")
    comp = composition(det, by="month", config=cfg)
    rich = richness(det, by="month", config=cfg)

    ms.to_csv(out / "morning_summary.csv", index=False)
    sp_month.to_csv(out / "species_phenology_month.csv", index=False)
    comp.to_csv(out / "composition_month.csv", index=False)
    rich.to_csv(out / "richness_month.csv", index=False)

    anchor = cfg["anchor"]
    if not sp_month.empty:
        months = sorted(sp_month["month"].unique())
        period = months[len(months) // 2]
        ax = plots.onset_by_species(sp_month, period, anchor=anchor)
        ax.figure.tight_layout(); ax.figure.savefig(out / "onset_by_species.png", dpi=130)
        plt.close(ax.figure)

    ax = plots.occupancy_heatmap(det, cfg)
    ax.figure.tight_layout(); ax.figure.savefig(out / "occupancy_heatmap.png", dpi=130)
    plt.close(ax.figure)

    ax = plots.composition_area(comp)
    ax.figure.tight_layout(); ax.figure.savefig(out / "composition_area.png", dpi=130)
    plt.close(ax.figure)

    # Empirical CDF of vocal activity (robust, full-distribution view). Empty when no
    # morning cleared min_detections_per_morning (e.g. a tiny first-day sample).
    ecdf = species_ecdf(det, by="month", config=cfg)
    if ecdf.empty:
        print("no morning met the ECDF detection floor "
              "(min_detections_per_morning); skipping ECDF outputs")
    else:
        eq = ecdf_quantiles(ecdf, by="month", quantiles=tuple(cfg.get("ecdf_quantiles", (0.05, 0.5, 0.95))))
        ecdf.to_csv(out / "species_ecdf_month.csv", index=False)
        eq.to_csv(out / "ecdf_quantiles_month.csv", index=False)

        # Consecutive-month timing shifts for the busiest species (descriptive)
        months = sorted(ecdf["month"].unique())
        top = (det.groupby("scientific_name").size().sort_values(ascending=False).head(6).index)
        shifts = [ecdf_distance(det, sp, "month", a, b, cfg)
                  for sp in top for a, b in zip(months[:-1], months[1:])]
        if shifts:
            import pandas as _pd
            _pd.DataFrame(shifts).to_csv(out / "ecdf_month_shifts.csv", index=False)

        if len(months) > 1:
            band = months[len(months) // 2]
            fig = plots.ecdf_small_multiples(ecdf, by="month", anchor=anchor, band_for=band)
            fig.savefig(out / "ecdf_by_month.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    if args.weather and "temperature_2m" in ms:
        ax = plots.onset_vs_temperature(ms, anchor=anchor)
        if ax is not None:
            ax.figure.tight_layout(); ax.figure.savefig(out / "onset_vs_temperature.png", dpi=130)
            plt.close(ax.figure)

    print(f"wrote CSVs + PNGs to {out}/")


if __name__ == "__main__":
    main()
