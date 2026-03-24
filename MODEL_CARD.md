# Model Card

## Model

- Name: `best_tiled.pt`
- Role: canonical promoted model served by the API
- Family: `ultralytics/yolo11n-obb`
- Model version: `dota_tiled_from_pretrained_local_v1`
- Runtime slot: `models/best_tiled.pt`

## Intended Use

This model is intended for:

- oriented object detection on aerial images
- demonstration of a lightweight local promotion workflow
- portfolio review of ML engineering and model traceability practices

It is not presented as a certified production model.

## Classes

- `plane`
- `ship`
- `small-vehicle`
- `large-vehicle`

## Dataset

- Dataset name: `dota_obb_4class_tiled`
- Dataset version: `local-v1`
- Train split id: `processed-split-train`
- Val split id: `processed-split-val`

See [data/README.md](data/README.md) for the folder layout and assumptions.

## Training Provenance

This promoted artifact was backfilled from the existing local run:

- Source run directory: `runs_obb/dota_tiled_from_pretrained`
- Source weights: `runs_obb/dota_tiled_from_pretrained/weights/best_tiled.pt`
- Source metrics file: `runs_obb/dota_tiled_from_pretrained/results.csv`

The full promoted metadata bundle lives under:

- [artifacts/promoted/manifest.json](artifacts/promoted/manifest.json)
- [artifacts/promoted/metrics.json](artifacts/promoted/metrics.json)
- [artifacts/promoted/params.json](artifacts/promoted/params.json)

## Metrics Summary

Backfilled from the best validation epoch recorded in `runs_obb/dota_tiled_from_pretrained/results.csv`:

- Best epoch: `17`
- Precision: `0.8732`
- Recall: `0.8354`
- mAP50: `0.8854`
- mAP50-95: `0.5744`

## Promotion Policy

- A run is promotable only when it has `manifest.json`, `metrics.json`, and `params.json`.
- Promotion copies the selected weights into the canonical slot `models/best_tiled.pt`.
- Promotion also copies the selected metadata bundle to `artifacts/promoted/`.

## Known Limitations

- The current promoted metadata was backfilled from historical local run artifacts.
- There is no external model registry.
- There is no automated drift detection or scheduled retraining.
- Training and evaluation are local-first workflows, not orchestrated pipelines.
