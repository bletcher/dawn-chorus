"""
Generate a synthetic BirdNET-Go-style SQLite DB with realistic dawn-chorus
structure, PLUS an optional mock Open-Meteo hourly weather cache so the full
weather join runs offline. Onset is made temperature-sensitive (warmer mornings
-> earlier singing) so weather_response() has a signal to recover.

Usage:
    python tools/make_synthetic_db.py demo.db demo_weather.csv
"""
import sqlite3, sys, math, random
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun

random.seed(7)
LAT, LON, TZ = 42.53, -72.53, "America/New_York"   # Montague, MA-ish
loc = LocationInfo(latitude=LAT, longitude=LON)

# name: (onset_min_from_dawn, span_min, peak_intensity, (start_doy, end_doy))
SPECIES = {
    "American Robin":         (-40, 150, 1.0, (1, 366)),
    "Song Sparrow":          (-10, 130, 0.8, (1, 366)),
    "Northern Cardinal":     (-25, 120, 0.7, (1, 366)),
    "Black-capped Chickadee": (10, 100, 0.6, (1, 366)),
    "Wood Thrush":           (-30, 110, 0.9, (120, 240)),
    "Red-eyed Vireo":         (20, 200, 0.7, (130, 250)),
    "Ovenbird":               (-5, 120, 0.6, (125, 235)),
    "Common Yellowthroat":    (35, 140, 0.5, (128, 245)),
}
SCI = {n: n.lower().replace(" ", "_") for n in SPECIES}


def day_temp(doy):
    """Seasonal mean temp (C) + a day-level anomaly that also drives onset."""
    seasonal = 8 + 14 * math.sin(2 * math.pi * (doy - 110) / 365)
    anom = random.gauss(0, 3.0)          # warm/cool spell
    return seasonal + anom, anom


def dawn_local(d):
    return sun(loc.observer, date=d, tzinfo=ZoneInfo(TZ))["dawn"]


def make(db_path, wx_path=None, start=date(2025, 3, 1), days=200, every_n_days=3):
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS notes")
    cur.execute("""CREATE TABLE notes (
        id INTEGER PRIMARY KEY, source_node TEXT, date TEXT, time TEXT,
        species_code TEXT, scientific_name TEXT, common_name TEXT,
        confidence REAL, latitude REAL, longitude REAL,
        threshold REAL, sensitivity REAL, clip_name TEXT)""")
    wx_rows = []
    rid = 0
    for k in range(0, days, every_n_days):
        d = start + timedelta(days=k)
        doy = d.timetuple().tm_yday
        dwn = dawn_local(d)
        tmean, anom = day_temp(doy)
        onset_shift = -1.4 * anom            # warmer-than-normal -> earlier onset
        seasonal_amp = 0.5 + 0.5 * math.exp(-((doy - 155) / 45) ** 2)

        for name, (onset0, span, inten, (s0, s1)) in SPECIES.items():
            if not (s0 <= doy <= s1):
                continue
            onset = onset0 + onset_shift
            n = int(inten * seasonal_amp * random.randint(25, 55))
            for _ in range(n):
                t = random.gauss(onset + span * 0.35, span * 0.35)
                if t < onset or t > onset + span:
                    continue
                dt = dwn + timedelta(minutes=t)
                conf = min(0.99, max(0.55, random.gauss(0.82, 0.1)))
                rid += 1
                cur.execute("INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, "demo", dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"),
                     SCI[name][:6], SCI[name], name, round(conf, 3),
                     LAT, LON, 0.5, 1.0, ""))
        # daytime false positives (must NOT define onset)
        for _ in range(random.randint(2, 6)):
            name = random.choice(list(SPECIES))
            dt = datetime.combine(d, datetime.min.time()) + timedelta(minutes=random.randint(0, 1439))
            rid += 1
            cur.execute("INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, "demo", dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"),
                 SCI[name][:6], SCI[name], name, round(random.uniform(0.55, 0.7), 3),
                 LAT, LON, 0.5, 1.0, ""))

        # mock hourly weather for this day (naive local, Open-Meteo cache format)
        if wx_path:
            for hour in range(0, 14):
                ts = datetime.combine(d, datetime.min.time()) + timedelta(hours=hour)
                diurnal = -4 * math.cos(2 * math.pi * (hour - 15) / 24)  # cool at dawn
                wx_rows.append(dict(
                    time=ts.strftime("%Y-%m-%dT%H:%M"),
                    temperature_2m=round(tmean + diurnal, 1),
                    cloud_cover=random.randint(0, 100),
                    precipitation=round(max(0, random.gauss(0, 0.3)), 1),
                    wind_speed_10m=round(abs(random.gauss(6, 3)), 1),
                    relative_humidity_2m=random.randint(55, 98)))
    con.commit()
    print(f"wrote {rid} detections to {db_path}")
    con.close()

    if wx_path:
        import csv
        with open(wx_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wx_rows[0].keys()))
            w.writeheader(); w.writerows(wx_rows)
        print(f"wrote {len(wx_rows)} hourly weather rows to {wx_path}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "demo.db"
    wx = sys.argv[2] if len(sys.argv) > 2 else None
    make(db, wx)
