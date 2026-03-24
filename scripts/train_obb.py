#!/usr/bin/env python3
"""
Runs a reproducible local YOLO OBB training job and records its artifacts.
"""

import argparse
from pathlib import Path
from typing import Any

from common import load_yaml_config, resolve_repo_path

from app.artifacts import (  # noqa: E402
    create_run_id,
    get_dataset_dir,
    get_git_commit,
    get_python_version,
    get_run_dir,
    get_ultralytics_version,
    save_run_bundle,
    summarize_results_csv,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiled YOLO OBB model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Dataset config YAML.")
    parser.add_argument(
        "--inference-config",
        default="configs/inference.yaml",
        help="Inference config YAML used to stamp the manifest.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    return parser.parse_args()


def load_dataset_manifest(data_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = get_dataset_dir(
        str(data_cfg["dataset"]["name"]),
        str(data_cfg["dataset"]["version"]),
    )
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest missing: {manifest_path}. Run `python scripts/prepare_data.py` first."
        )
    from app.artifacts import read_json  # noqa: E402

    return read_json(manifest_path)


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml_config(resolve_repo_path(args.config))
    data_cfg = load_yaml_config(resolve_repo_path(args.data_config))
    inference_cfg = load_yaml_config(resolve_repo_path(args.inference_config))
    dataset_manifest = load_dataset_manifest(data_cfg)
    dataset_manifest_path = (
        get_dataset_dir(
            str(data_cfg["dataset"]["name"]),
            str(data_cfg["dataset"]["version"]),
        )
        / "dataset_manifest.json"
    )

    run_id = args.run_id or create_run_id(str(train_cfg["run"]["prefix"]))
    run_dir = get_run_dir(run_id)
    training_output_name = "ultralytics_train"
    training_output_dir = run_dir / training_output_name

    initial_weights = str(train_cfg["run"]["initial_weights"])
    training_params = dict(train_cfg["training"])
    training_params.update(
        {
            "task": "obb",
            "data": dataset_manifest["ultralytics_data_file"],
            "project": str(run_dir),
            "name": training_output_name,
            "exist_ok": True,
            "verbose": True,
            "plots": True,
            "save": True,
        }
    )

    from ultralytics import YOLO  # noqa: E402

    model = YOLO(initial_weights)
    model.train(**training_params)

    best_weights_path = training_output_dir / "weights" / "best.pt"
    if not best_weights_path.exists():
        raise FileNotFoundError(f"Best weights not found after training: {best_weights_path}")

    results_csv = training_output_dir / "results.csv"
    metrics_summary = summarize_results_csv(results_csv) if results_csv.exists() else {}
    metrics = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "metrics_summary": metrics_summary,
        "results_csv": str(results_csv),
        "training_output_dir": str(training_output_dir),
    }
    params = {
        "run": train_cfg["run"],
        "training": train_cfg["training"],
        "dataset": data_cfg["dataset"],
        "inference": inference_cfg["inference"],
        "resolved": {
            "data_manifest": dataset_manifest,
            "training_output_dir": str(training_output_dir),
            "initial_weights": initial_weights,
        },
    }
    manifest = {
        "run_id": run_id,
        "model_name": best_weights_path.name,
        "model_family": f"ultralytics/{Path(initial_weights).stem}",
        "model_version": run_id,
        "dataset_name": dataset_manifest["dataset_name"],
        "dataset_version": dataset_manifest["dataset_version"],
        "train_split_id": dataset_manifest["train_split_id"],
        "val_split_id": dataset_manifest["val_split_id"],
        "ultralytics_version": get_ultralytics_version(),
        "python_version": get_python_version(),
        "git_commit": get_git_commit(),
        "training_date": utc_now_iso(),
        "tile_size": inference_cfg["inference"]["tile_size"],
        "overlap": inference_cfg["inference"]["overlap"],
        "confidence_threshold": inference_cfg["inference"]["confidence_threshold"],
        "metrics_summary": metrics_summary,
        "promoted_to_service": False,
        "source_model_path": str(best_weights_path),
        "training_output_dir": str(training_output_dir),
        "dataset_manifest_path": str(dataset_manifest_path),
    }

    save_run_bundle(run_id, manifest, metrics, params)

    print(f"[OK] Training artifacts saved under {run_dir}")
    print(f"[OK] Best weights: {best_weights_path}")
    print(f"[OK] Run id: {run_id}")


if __name__ == "__main__":
    main()
