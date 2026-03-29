# Architecture

## Purpose

This project serves oriented object detection on aerial imagery while keeping the ML workflow reproducible and reviewable from plain Python modules.

## Runtime flow

1. `app/main.py` accepts uploads, authenticates `X-API-Key`, applies rate limits, propagates `X-Request-ID`, and rejects work when inference slots are saturated.
2. `app/inference.py` loads the promoted YOLO OBB model and runs tiled inference under a thread-safe detector lock.
3. `app/utils.py` handles tile geometry, reprojection, and global NMS.
4. `app/artifacts.py` exposes promoted model metadata for `/model-info`.
5. `app/observability.py` emits structured request logs with request ID, client IP, API key fingerprint, and model version.

## Training and dataset workflow

1. `scripts/prepare_data.py` builds or validates the tiled processed dataset.
2. `app/dataset_tiling.py` is the source of truth for parsing DOTA labels, generating training tiles, clipping OBBs to tile bounds, normalizing YOLO OBB labels, and deterministic empty-tile retention.
3. `scripts/train_obb.py` runs training and records a manifest, metrics, and params bundle.
4. `scripts/evaluate_obb.py` refreshes validation metrics for a run.
5. `scripts/promote_model.py` copies the selected weights into `models/best_tiled.pt` and updates `artifacts/promoted/`.

## Operational boundaries

Implemented now:

- authenticated protected routes
- in-memory rate limiting
- request correlation IDs
- fail-fast in-flight capacity guard
- promoted model metadata and local artifact lineage
- config-driven dataset tiling outside notebooks

Explicitly not implemented yet:

- distributed job queue for large asynchronous requests
- external model registry
- distributed rate-limit backend
- OAuth or centralized identity provider
- remote dataset versioning

## Notebook role

The notebooks under `notebooks/` are not the operational source of truth. They are kept only for:

- exploratory data analysis
- experiment notes
- demo-friendly walkthroughs

If logic becomes required for runtime, training, or reproducibility, it must live in `app/` or `scripts/`.
