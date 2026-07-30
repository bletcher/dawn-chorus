"""
Seed the local DB with the current recordings as one site, for development.

    cd server && python seed.py

Loads ../data/results (BirdNET output) via dawnchorus and inserts the detections under a
demo site. Idempotent: re-running replaces that site's detections. Prints the api_key the
first time the site is created (used for uploads).
"""
import datetime as dt
import secrets

import pandas as pd
from sqlalchemy import delete, select

import dawnchorus as dc
from app import Base, Detection, SessionLocal, Site, _hash, engine

LAT, LON, TZ = 42.53, -72.53, "America/New_York"
SLUG, NAME = "montague", "Montague, MA (demo)"

Base.metadata.create_all(engine)

det = dc.load_birdnet_analyzer("../data/results", min_confidence=0.1,
                               latitude=LAT, longitude=LON, tz=TZ)
print(f"loaded {len(det)} detections from ../data/results")

with SessionLocal() as db:
    site = db.scalar(select(Site).where(Site.slug == SLUG))
    if not site:
        key = secrets.token_urlsafe(24)
        site = Site(slug=SLUG, name=NAME, latitude=LAT, longitude=LON, tz=TZ,
                    api_key_hash=_hash(key), created_at=dt.datetime.utcnow())
        db.add(site); db.commit()
        print(f"created site '{SLUG}'  |  api_key: {key}")

    db.execute(delete(Detection).where(Detection.site_id == site.id))
    objs = []
    for r in det.itertuples():
        d = pd.Timestamp(r.datetime).to_pydatetime()
        if d.tzinfo is not None:
            d = d.replace(tzinfo=None)                      # store naive station-local wall-clock
        objs.append(Detection(site_id=site.id, dt=d, scientific_name=str(r.scientific_name),
                              common_name=str(r.common_name), confidence=float(r.confidence),
                              model_version="BirdNET_V2.4"))
    db.add_all(objs); db.commit()
    print(f"inserted {len(objs)} detections for '{SLUG}'")
