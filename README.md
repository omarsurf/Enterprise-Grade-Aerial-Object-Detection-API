# Production-Ready Aerial OBB Detection API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.135.2-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/YOLO-v11_OBB-00FFFF?style=for-the-badge" alt="YOLO"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<p align="center">
  <strong>Production-oriented FastAPI service for oriented object detection on aerial imagery.</strong>
</p>

This repository is centered on Python modules and workflow scripts, not notebooks. The operational source of truth lives here:

| Area | Source of truth |
|---|---|
| API surface | `app/main.py` |
| Tiled inference runtime | `app/inference.py` |
| Dataset tiling pipeline | `app/dataset_tiling.py` |
| Data build / validate workflow | `scripts/prepare_data.py` |
| Train / evaluate / promote workflow | `scripts/train_obb.py`, `scripts/evaluate_obb.py`, `scripts/promote_model.py` |
| Artifact metadata and traceability | `app/artifacts.py`, `MODEL_CARD.md` |

Notebooks remain in `notebooks/`, but only as lightweight EDA, experiment notes, and demo walkthroughs.

## Why OBB Detection?

Traditional bounding boxes fail on rotated aircraft, ships, and vehicles. Oriented Bounding Boxes provide tighter geometry and better localization for aerial scenes where object pose matters.

```text
Traditional Box (Imprecise)          Oriented Box (Precise)
┌─────────────────┐                      ╱╲
│       ✈         │                    ╱  ╲
│                 │         vs        ╱ ✈  ╲
│                 │                   ╲    ╱
└─────────────────┘                    ╲  ╱
                                        ╲╱
```

## What "Production-Ready" Means Here

The label is now scoped explicitly instead of implied.

| Concern | Implemented now | Next hardening step |
|---|---|---|
| Runtime isolation | Docker image, non-root user, healthcheck | multi-worker deployment and autoscaling |
| API protection | `X-API-Key` auth on protected routes | external secret manager / OAuth |
| Abuse control | in-memory rate limiting with `slowapi` | distributed limiter backed by Redis |
| Request tracing | `X-Request-ID` propagation + structured logs | shipping logs to central observability stack |
| Concurrency guard | fail-fast in-flight prediction semaphore | job queue for large asynchronous workloads |
| Model traceability | promoted manifest, params, metrics bundle | external model registry |
| Dataset reproducibility | config-driven tiling build + dataset manifest | DVC or remote dataset versioning |

## Model Provenance

The headline metrics are tied to the promoted local artifact, not presented as a certified production benchmark.

This repository intentionally focuses on a constrained training scope: a four-class slice of DOTA (`local-v1`) and the lightweight `yolo11n-obb` variant. The goal was to keep full end-to-end iteration practical on CPU-first local hardware, avoid turning each experiment into a long Colab session, and still demonstrate the complete `prepare -> train -> evaluate -> promote -> serve` workflow. If the objective were to push raw model capability further, the next levers are straightforward: add more target classes, train on a larger image set, and move to a stronger YOLO OBB backbone. The surrounding pipeline is designed so those upgrades are an extension path, not a redesign.

| Item | Value |
|---|---|
| Dataset version | `local-v1` |
| Train images | `125` |
| Validation images | `32` |
| Classes | `plane`, `ship`, `small-vehicle`, `large-vehicle` |
| Served model | `models/best_tiled.pt` |
| Model family | `ultralytics/yolo11n-obb` |
| Promoted metrics | Precision `0.8732`, Recall `0.8354`, mAP50 `0.8854`, mAP50-95 `0.5744` |

See `MODEL_CARD.md` and `artifacts/promoted/` for the promoted bundle.

## Architecture

High-level architecture is documented in `docs/architecture.md`.

```text
client
  -> FastAPI request validation
  -> API key auth + rate limit + request ID logging
  -> image decode
  -> tiled OBB inference
  -> global merge / NMS
  -> JSON response or annotated output
```

## Quick Start

### 1. Install the CPU API environment

```bash
git clone https://github.com/omarsurf/Enterprise-Grade-Aerial-Object-Detection-API.git aerial-obb-api
cd aerial-obb-api
make install
```

Manual equivalent:

```bash
pip install -r requirements/api-cpu.txt
pip install -e . --no-deps
```

### 2. Run the API

```bash
export AERIAL_API_KEYS=local-dev-key
make run
```

Swagger UI: `http://localhost:8000/docs`

### 3. Authenticated prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: local-dev-key" \
  -H "X-Request-ID: demo-001" \
  -F "file=@satellite_image.png"
```

### 4. Docker / Compose

```bash
docker build -t aerial-obb-api .
docker run -e AERIAL_API_KEYS=local-dev-key -p 8000:8000 aerial-obb-api
```

`docker-compose.yml` already includes local defaults for:

- `AERIAL_API_KEYS`
- `RATE_LIMIT_PREDICT`
- `RATE_LIMIT_PREDICT_AND_SAVE`
- `RATE_LIMIT_MODEL_INFO`
- `MAX_INFLIGHT_PREDICTIONS`
- `BUSY_RETRY_AFTER_SECONDS`

## API Surface

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | `GET` | no | service + model readiness |
| `/model-info` | `GET` | yes | promoted model metadata and metrics |
| `/predict` | `POST` | yes | tiled OBB detection to JSON |
| `/predict-and-save` | `POST` | yes | tiled OBB detection + annotated image path |

Protected endpoints require `X-API-Key`. All routes accept and return `X-Request-ID`.

Additional runtime behavior:

- rate limiting via `slowapi`
- fail-fast saturation protection when all inference slots are busy
- structured request logging without exposing raw API keys

## Reproducible ML Workflow

```text
prepare-data -> train -> evaluate -> promote -> serve
```

```bash
make train-env
make prepare-data
make train
make evaluate
make promote-model
make smoke-inference
```

### `prepare-data` behavior

`scripts/prepare_data.py` now supports:

```bash
python scripts/prepare_data.py --config configs/data.yaml --mode build
python scripts/prepare_data.py --config configs/data.yaml --mode validate
```

`build` reconstructs the tiled processed split from filtered DOTA-style source labels. `validate` checks an already-built processed split and refreshes the dataset manifest.

## Project Structure

```text
aerial-obb-api/
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── dataset_tiling.py
│   ├── security.py
│   ├── observability.py
│   ├── artifacts.py
│   ├── config.py
│   └── schemas.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_obb.py
│   ├── evaluate_obb.py
│   ├── promote_model.py
│   └── smoke_inference.py
├── configs/
├── artifacts/
├── tests/
├── notebooks/
├── docs/
└── MODEL_CARD.md
```

## Notebook Policy

- notebooks are exploratory only
- outputs are stripped before commit
- `google.colab` glue and `!pip install` cells are removed
- operational logic must remain importable from Python modules

## Development

```bash
make dev
make lint
make test
make validate-artifacts
```

Pre-commit includes Ruff and `nbstripout` to keep notebooks lightweight.
