#!/usr/bin/env python3
"""
Evaluates a trained OBB model on the fixed validation split and refreshes run metrics.
"""

import argparse
from pathlib import Path
from typing import Any

from common import load_yaml_config, resolve_repo_path

from app.artifacts import (  # noqa: E402
    get_dataset_dir,
    get_run_dir,
    read_json,
    read_latest_run_id,
    summarize_results_csv,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing YOLO OBB run.")
    parser.add_argument(
        "--run-id", default=None, help="Run id to evaluate. Defaults to the latest run."
    )
    parser.add_argument("--data-config", default="configs/data.yaml", help="Dataset config YAML.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional weights path. Defaults to the run manifest source_model_path.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Optional existing results.csv to backfill metrics without running validation.",
    )
    return parser.parse_args()


def normalize_results_dict(results_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in results_dict.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def build_summary_from_results(results_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": round(float(results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or read_latest_run_id()
    if not run_id:
        raise ValueError("No run id provided and no latest run id recorded yet.")

    run_dir = get_run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.json"

    if not manifest_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"Run bundle incomplete for {run_id}: {run_dir}")

    manifest = read_json(manifest_path)
    read_json(params_path)

    if args.results_csv:
        results_csv = resolve_repo_path(args.results_csv)
        metrics_summary = summarize_results_csv(results_csv)
        metrics = {
            "run_id": run_id,
            "evaluated_at": utc_now_iso(),
            "metrics_summary": metrics_summary,
            "results_csv": str(results_csv),
            "evaluation_mode": "backfill_from_results_csv",
        }
    else:
        data_cfg = load_yaml_config(resolve_repo_path(args.data_config))
        dataset_dir = get_dataset_dir(
            str(data_cfg["dataset"]["name"]),
            str(data_cfg["dataset"]["version"]),
        )
        dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")
        weights_path = Path(args.weights or manifest["source_model_path"])

        from ultralytics import YOLO  # noqa: E402

        model = YOLO(str(weights_path))
        evaluation_dir = run_dir / "evaluation"
        metrics_obj = model.val(
            task="obb",
            data=dataset_manifest["ultralytics_data_file"],
            split="val",
            project=str(run_dir),
            name="evaluation",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        results_dict = normalize_results_dict(getattr(metrics_obj, "results_dict", {}) or {})
        metrics_summary = build_summary_from_results(results_dict)
        metrics = {
            "run_id": run_id,
            "evaluated_at": utc_now_iso(),
            "metrics_summary": metrics_summary,
            "results_dict": results_dict,
            "weights_path": str(weights_path),
            "evaluation_output_dir": str(evaluation_dir),
            "evaluation_mode": "ultralytics_val",
        }

    manifest["metrics_summary"] = metrics_summary
    write_json(manifest_path, manifest)
    write_json(metrics_path, metrics)

    print(f"[OK] Evaluation metrics updated for {run_id}")
    print(f"[OK] Metrics summary: {metrics_summary}")


if __name__ == "__main__":
    main()
