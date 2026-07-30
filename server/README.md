# dawn-chorus API

Multi-site detection store + dashboard payloads. FastAPI + SQLAlchemy, reusing the
`dawnchorus` analysis engine. SQLite locally; PostgreSQL (RDS) in production — set
`DATABASE_URL`.

## Run locally

```bash
pip install -r requirements.txt
pip install -e ..                 # the dawnchorus analysis engine
python seed.py                    # loads ../data/results as the demo site "montague"
uvicorn app:app --port 8001       # http://127.0.0.1:8001/docs
```

## Endpoints

| method | path | notes |
|---|---|---|
| `POST` | `/sites` | create a site → `{slug, api_key}` (key shown once) |
| `GET`  | `/sites` | list sites — powers the site selector |
| `POST` | `/sites/{slug}/detections` | upload detections; header `X-API-Key` |
| `GET`  | `/sites/{slug}/data` | render-ready dashboard payload for a site |

`GET /sites/{slug}/data?min_conf=0.5&label_min_conf=0.25` returns the same JSON the static
dashboard embeds (meta / summary / counts / day_keys / dets) — minus audio, since recordings
stay on contributors' machines. The frontend does the time-scope aggregation and ECDF.

## Data model

- **sites** — `slug, name, lat, lon, tz, api_key_hash`
- **detections** — `site_id, dt (naive local), scientific_name, common_name, confidence, model_version`, indexed on `(site_id, dt)`

## Production (planned)

RDS Postgres + this app as a container on AWS App Runner; frontend + the BirdNET model on the
existing S3/CloudFront. `DATABASE_URL` and any keys via Secrets Manager.
