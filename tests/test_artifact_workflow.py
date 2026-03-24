"""
Additional tests for the local artifact workflow and promotion helpers.
"""

import csv
import re
from pathlib import Path
from types import SimpleNamespace

from app import artifacts


def patch_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    """Redirect artifact paths to a temporary sandbox."""
    artifacts_dir = tmp_path / "artifacts"
    promoted_dir = artifacts_dir / "promoted"
    datasets_dir = artifacts_dir / "datasets"
    latest_run_id_path = artifacts_dir / ".latest_run_id"
    model_path = tmp_path / "models" / "best_tiled.pt"

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "PROMOTED_MANIFEST_PATH", promoted_dir / "manifest.json")
    monkeypatch.setattr(artifacts, "PROMOTED_METRICS_PATH", promoted_dir / "metrics.json")
    monkeypatch.setattr(artifacts, "PROMOTED_PARAMS_PATH", promoted_dir / "params.json")
    monkeypatch.setattr(artifacts, "DATASET_ARTIFACTS_DIR", datasets_dir)
    monkeypatch.setattr(artifacts, "LATEST_RUN_ID_PATH", latest_run_id_path)
    monkeypatch.setattr(artifacts, "MODEL_PATH", model_path)


def test_slugify_and_create_run_id_format():
    """Run ids should preserve a stable slug prefix."""
    assert artifacts.slugify("DOTA tiled / local v1") == "dota-tiled-local-v1"

    run_id = artifacts.create_run_id("DOTA tiled / local v1")

    assert re.fullmatch(r"dota-tiled-local-v1_\d{8}_\d{6}", run_id)


def test_get_git_commit_handles_success_and_failure(monkeypatch):
    """Git commit discovery should be explicit in both happy and fallback paths."""

    class CompletedProcess:
        stdout = "abc123\n"

    monkeypatch.setattr(artifacts.subprocess, "run", lambda *args, **kwargs: CompletedProcess())
    assert artifacts.get_git_commit() == "abc123"

    def fail_run(*args, **kwargs):
        raise artifacts.subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(artifacts.subprocess, "run", fail_run)
    assert artifacts.get_git_commit() == "unknown"


def test_get_python_and_ultralytics_version(monkeypatch):
    """Version helpers should expose runtime versions."""
    dummy_ultralytics = SimpleNamespace(__version__="9.9.9")
    monkeypatch.setitem(artifacts.sys.modules, "ultralytics", dummy_ultralytics)

    assert re.fullmatch(r"\d+\.\d+\.\d+", artifacts.get_python_version())
    assert artifacts.get_ultralytics_version() == "9.9.9"


def test_save_run_bundle_and_latest_run_id(monkeypatch, tmp_path):
    """Saving a run bundle should persist all JSON files and the latest run id."""
    patch_artifact_paths(monkeypatch, tmp_path)

    run_dir = artifacts.save_run_bundle(
        run_id="run-001",
        manifest={"run_id": "run-001", "model_name": "best.pt"},
        metrics={"mAP50": 0.8},
        params={"epochs": 50},
    )

    assert run_dir == tmp_path / "artifacts" / "run-001"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "params.json").exists()
    assert artifacts.read_latest_run_id() == "run-001"


def test_json_and_path_helpers(monkeypatch, tmp_path):
    """JSON and path helpers should be stable and filesystem-friendly."""
    patch_artifact_paths(monkeypatch, tmp_path)
    payload = {"status": "ok"}
    json_path = tmp_path / "sample.json"

    artifacts.write_json(json_path, payload)

    assert artifacts.read_json(json_path) == payload
    assert artifacts.read_json_if_exists(json_path) == payload
    assert artifacts.read_json_if_exists(tmp_path / "missing.json") is None
    assert artifacts.get_run_dir("run-123") == tmp_path / "artifacts" / "run-123"
    assert artifacts.get_dataset_dir("dota tiled", "local-v1").name == "dota-tiled_local-v1"


def test_summarize_results_csv_uses_best_map_row(tmp_path):
    """Metrics summary should be extracted from the best validation row."""
    results_csv_path = tmp_path / "results.csv"
    fieldnames = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    rows = [
        {
            "epoch": "0",
            "metrics/precision(B)": "0.70",
            "metrics/recall(B)": "0.60",
            "metrics/mAP50(B)": "0.75",
            "metrics/mAP50-95(B)": "0.40",
        },
        {
            "epoch": "2",
            "metrics/precision(B)": "0.80",
            "metrics/recall(B)": "0.72",
            "metrics/mAP50(B)": "0.88",
            "metrics/mAP50-95(B)": "0.55",
        },
    ]

    with results_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = artifacts.summarize_results_csv(results_csv_path)

    assert summary["best_epoch"] == 2
    assert summary["mAP50-95"] == 0.55
    assert summary["precision"] == 0.8


def test_load_promoted_bundle_and_promote_run_artifacts(monkeypatch, tmp_path):
    """Promotion should copy weights and publish the promoted metadata bundle."""
    patch_artifact_paths(monkeypatch, tmp_path)

    run_id = "run-002"
    run_dir = artifacts.get_run_dir(run_id)
    run_dir.mkdir(parents=True)
    source_model_path = tmp_path / "runs" / "best.pt"
    source_model_path.parent.mkdir(parents=True)
    source_model_path.write_bytes(b"weights")

    manifest = {
        "run_id": run_id,
        "model_name": "best.pt",
        "model_family": "ultralytics/yolo11n-obb",
        "model_version": run_id,
        "dataset_name": "dota_obb_4class_tiled",
        "dataset_version": "local-v1",
        "train_split_id": "train-split",
        "val_split_id": "val-split",
        "ultralytics_version": "8.3.0",
        "python_version": "3.11.9",
        "git_commit": "abc123",
        "training_date": "2026-03-24T00:00:00+00:00",
        "tile_size": 1024,
        "overlap": 200,
        "confidence_threshold": 0.55,
        "metrics_summary": {"mAP50-95": 0.55},
        "promoted_to_service": False,
        "source_model_path": str(source_model_path),
    }
    metrics = {"metrics_summary": {"mAP50-95": 0.55}}
    params = {"training": {"epochs": 50}}

    artifacts.write_json(run_dir / "manifest.json", manifest)
    artifacts.write_json(run_dir / "metrics.json", metrics)
    artifacts.write_json(run_dir / "params.json", params)

    promoted_manifest = artifacts.promote_run_artifacts(run_id)
    bundle = artifacts.load_promoted_bundle()

    assert promoted_manifest["promoted_to_service"] is True
    assert promoted_manifest["model_name"] == "best_tiled.pt"
    assert artifacts.MODEL_PATH.exists()
    assert bundle["manifest"]["run_id"] == run_id
    assert bundle["metrics"] == metrics
    assert bundle["params"] == params
