# Data Layout

This repository uses a lightweight local data workflow. Large files stay out of Git; the repo only keeps structure, manifests, and documentation.

## Folder Roles

- `data/raw/`
  Original local DOTA images and annotations.
- `data/filtered/`
  Filtered subset used to keep only the four target classes.
- `data/processed/`
  Prepared train/validation split used by the reproducible workflow.
- `data/interim/`
  Temporary or sample subsets used during local experimentation.
- `data/inference_test/`
  Local-only inference playground images. This folder is not tracked.

## Active Dataset Assumption

The current lightweight workflow assumes:

- dataset name: `dota_obb_4class_tiled`
- dataset version: `local-v1`
- source train images: `data/filtered/split/train/images`
- source train labels: `data/filtered/split/train/labelTxt`
- source val images: `data/filtered/split/val/images`
- source val labels: `data/filtered/split/val/labelTxt`
- train images: `data/processed/split/train/images`
- train labels: `data/processed/split/train/labelTxt`
- val images: `data/processed/split/val/images`
- val labels: `data/processed/split/val/labelTxt`

The active classes are:

- `plane`
- `ship`
- `small-vehicle`
- `large-vehicle`

## Source of Truth

For operational workflow purposes, the source of truth is:

- [configs/data.yaml](../configs/data.yaml)
- [scripts/prepare_data.py](../scripts/prepare_data.py)
- generated manifests under `artifacts/datasets/`

Notebooks are exploratory only and are not the operational source of truth.

## Rebuild / Validate

Build or validate the local tiled dataset and emit a manifest:

```bash
make prepare-data
```

This command:

- builds processed tiled splits from filtered DOTA-style source labels
- checks that train/val images and labels exist
- validates 1:1 image/label stem alignment
- attempts to create `labels` aliases next to image folders for Ultralytics compatibility
- writes:
  - `artifacts/datasets/<dataset_id>/dataset_manifest.json`
  - `artifacts/datasets/<dataset_id>/ultralytics_data.yaml`

Manual modes:

```bash
python scripts/prepare_data.py --config configs/data.yaml --mode build
python scripts/prepare_data.py --config configs/data.yaml --mode validate
```

## Git Policy

- Large raw/processed data stays outside Git.
- Logic, configs, and small manifests stay in Git.
- If the project grows into frequent retraining or multi-dataset management, DVC would be the next logical addition.
