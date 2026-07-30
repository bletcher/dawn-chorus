"""
Dawn-chorus API — multi-site detection store + dashboard payload.

    uvicorn app:app --reload            # http://127.0.0.1:8001/docs

Endpoints:
  POST /sites                      create a site -> {slug, api_key}
  GET  /sites                      list sites (for the site selector)
  POST /sites/{slug}/detections    upload detections (header X-API-Key)
  GET  /sites/{slug}/data          dashboard payload for a site (public)

Local dev uses SQLite (DATABASE_URL default); production points DATABASE_URL at RDS Postgres.
The heavy analysis is the tested `dawnchorus` library, called in payload.build_payload.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import secrets

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (DateTime, Float, ForeignKey, String, create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from payload import build_payload

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dawnchorus.db")
engine = create_engine(DATABASE_URL,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    tz: Mapped[str] = mapped_column(String)
    api_key_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"), index=True)
    dt: Mapped[dt.datetime] = mapped_column(DateTime, index=True)     # naive, station-local
    scientific_name: Mapped[str] = mapped_column(String)
    common_name: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String, default="")


Base.metadata.create_all(engine)

app = FastAPI(title="dawn-chorus API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _hash(k: str) -> str:
    return hashlib.sha256(k.encode()).hexdigest()


class SiteIn(BaseModel):
    slug: str
    name: str
    latitude: float
    longitude: float
    tz: str


class DetectionIn(BaseModel):
    dt: dt.datetime
    scientific_name: str
    common_name: str
    confidence: float


@app.post("/sites")
def create_site(s: SiteIn):
    with SessionLocal() as db:
        if db.scalar(select(Site).where(Site.slug == s.slug)):
            raise HTTPException(409, "slug already exists")
        key = secrets.token_urlsafe(24)
        db.add(Site(slug=s.slug, name=s.name, latitude=s.latitude, longitude=s.longitude,
                    tz=s.tz, api_key_hash=_hash(key), created_at=dt.datetime.utcnow()))
        db.commit()
        return {"slug": s.slug, "api_key": key}      # shown once — the contributor's upload key


@app.get("/sites")
def list_sites():
    with SessionLocal() as db:
        out = []
        for site in db.scalars(select(Site)).all():
            n = db.scalar(select(func.count(Detection.id)).where(Detection.site_id == site.id)) or 0
            lo, hi = db.execute(select(func.min(Detection.dt), func.max(Detection.dt))
                                .where(Detection.site_id == site.id)).first()
            out.append({"slug": site.slug, "name": site.name, "lat": site.latitude,
                        "lon": site.longitude, "tz": site.tz, "n_detections": n,
                        "first": str(lo)[:10] if lo else None, "last": str(hi)[:10] if hi else None})
        return out


@app.post("/sites/{slug}/detections")
def upload_detections(slug: str, dets: list[DetectionIn], x_api_key: str = Header(None)):
    with SessionLocal() as db:
        site = db.scalar(select(Site).where(Site.slug == slug))
        if not site:
            raise HTTPException(404, "unknown site")
        if not x_api_key or _hash(x_api_key) != site.api_key_hash:
            raise HTTPException(401, "invalid API key for this site")
        db.add_all([Detection(site_id=site.id, dt=d.dt, scientific_name=d.scientific_name,
                              common_name=d.common_name, confidence=d.confidence) for d in dets])
        db.commit()
        return {"inserted": len(dets)}


@app.get("/sites/{slug}/data")
def site_data(slug: str, min_conf: float = 0.5, label_min_conf: float = 0.25):
    with SessionLocal() as db:
        site = db.scalar(select(Site).where(Site.slug == slug))
        if not site:
            raise HTTPException(404, "unknown site")
        rows = db.execute(select(Detection.dt, Detection.scientific_name, Detection.common_name,
                                 Detection.confidence).where(Detection.site_id == site.id)).all()
    if not rows:
        raise HTTPException(404, "no detections for this site yet")
    df = pd.DataFrame(rows, columns=["datetime", "scientific_name", "common_name", "confidence"])
    return build_payload(df, site.latitude, site.longitude, site.tz, min_conf, label_min_conf)
