"""
Helpers for lightweight local MLOps artifacts and promoted model metadata.
"""

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import (
    ARTIFACTS_DIR,
    DATASET_ARTIFACTS_DIR,
    MODEL_PATH,
    PROMOTED_MANIFEST_PATH,
    PROMOTED_METRICS_PATH,
    PROMOTED_PARAMS_PATH,
    logger,
)

LATEST_RUN_ID_PATH = ARTIFACTS_DIR / ".latest_run_id"

REQUIRED_MODEL_MANIFEST_KEYS = [
    "run_id",
    "model_name",
    "model_family",
    "model_version",
    "dataset_name",
    "dataset_version",
    "train_split_id",
    "val_split_id",
    "ultralytics_version",
    "python_version",
    "git_commit",
    "training_date",
    "tile_size",
    "overlap",
    "confidence_threshold",
    "metrics_summary",
    "promoted_to_service",
]

REQUIRED_DATASET_MANIFEST_KEYS = [
    "dataset_name",
    "dataset_version",
    "dataset_root",
    "train_images",
    "train_labels",
    "val_images",
    "val_labels",
    "class_names",
]


def utc_now_iso() -> str:
    """Returns an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    """Builds a stable filesystem-friendly slug."""
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("-")

    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "artifact"


def ensure_artifact_layout() -> None:
    """Creates the local artifact folders used by the workflow."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMOTED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_run_id(prefix: str = "obb") -> str:
    """Creates a time-based run id for local training/evaluation runs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{slugify(prefix)}_{timestamp}"


def get_git_commit() -> str:
    """Returns the current git commit, or 'unknown' outside a git repo."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def get_python_version() -> str:
    """Returns the active Python version."""
    return sys.version.split()[0]


def get_ultralytics_version() -> str:
    """Returns the installed ultralytics version when available."""
    try:
        import ultralytics  # pylint: disable=import-outside-toplevel
    except ImportError:
        return "unknown"
    return getattr(ultralytics, "__version__", "unknown")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Loads a JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    """Loads a JSON payload when the file exists."""
    if not path.exists():
        return None
    return read_json(path)


def write_latest_run_id(run_id: str) -> None:
    """Persists the most recent workflow run id."""
    ensure_artifact_layout()
    LATEST_RUN_ID_PATH.write_text(run_id + "\n", encoding="utf-8")


def read_latest_run_id() -> Optional[str]:
    """Reads the most recent workflow run id."""
    if not LATEST_RUN_ID_PATH.exists():
        return None
    return LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip() or None


def get_run_dir(run_id: str) -> Path:
    """Returns the directory used for a model run."""
    return ARTIFACTS_DIR / run_id


def get_dataset_dir(dataset_name: str, dataset_version: str) -> Path:
    """Returns the directory used for dataset manifests."""
    dataset_id = f"{slugify(dataset_name)}_{slugify(dataset_version)}"
    return DATASET_ARTIFACTS_DIR / dataset_id


def validate_required_keys(payload: dict[str, Any], required_keys: list[str]) -> list[str]:
    """Returns the list of missing required keys for a JSON payload."""
    return [key for key in required_keys if key not in payload]


def summarize_results_csv(results_csv_path: Path) -> dict[str, Any]:
    """Extracts the best validation summary from an Ultralytics results CSV."""
    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Le fichier results.csv est vide: {results_csv_path}")

    best_row = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0.0)))

    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "precision": round(float(best_row.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(best_row.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(best_row.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(best_row.get("metrics/mAP50-95(B)", 0.0)), 4),
        "source_results_csv": str(results_csv_path),
    }


def load_promoted_bundle() -> dict[str, Optional[dict[str, Any]]]:
    """Loads the promoted manifest, metrics, and params bundle."""
    return {
        "manifest": read_json_if_exists(PROMOTED_MANIFEST_PATH),
        "metrics": read_json_if_exists(PROMOTED_METRICS_PATH),
        "params": read_json_if_exists(PROMOTED_PARAMS_PATH),
    }


def build_model_info(model_loaded: bool) -> dict[str, Any]:
    """Builds the runtime model-info payload exposed by the API."""
    bundle = load_promoted_bundle()
    manifest = bundle["manifest"] or {}
    metrics = bundle["metrics"] or {}

    metrics_summary = manifest.get("metrics_summary") or metrics.get("metrics_summary") or {}
    manifest_available = bool(manifest)

    payload = {
        "model_loaded": model_loaded,
        "manifest_available": manifest_available,
        "model_name": manifest.get("model_name") or MODEL_PATH.name,
        "model_path": str(MODEL_PATH),
        "model_version": manifest.get("model_version"),
        "run_id": manifest.get("run_id"),
        "dataset_version": manifest.get("dataset_version"),
        "trained_at": manifest.get("training_date"),
        "metrics_summary": metrics_summary,
        "message": None,
    }

    if not manifest_available:
        payload["message"] = (
            "Promoted model manifest not found. Run `make promote-model` to publish metadata."
        )

    return payload


def save_run_bundle(
    run_id: str,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    params: dict[str, Any],
) -> Path:
    """Writes the manifest, metrics, and params bundle for a run."""
    run_dir = get_run_dir(run_id)
    ensure_artifact_layout()
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "params.json", params)
    write_latest_run_id(run_id)
    return run_dir


def promote_run_artifacts(
    run_id: str,
    source_model_path: Optional[Path] = None,
    target_model_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Promotes a run bundle and copies its weights to the canonical served model."""
    target_path = target_model_path or MODEL_PATH
    run_dir = get_run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable pour le run {run_id}: {manifest_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics introuvables pour le run {run_id}: {metrics_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Params introuvables pour le run {run_id}: {params_path}")

    manifest = read_json(manifest_path)
    metrics = read_json(metrics_path)
    params = read_json(params_path)

    missing_keys = validate_required_keys(manifest, REQUIRED_MODEL_MANIFEST_KEYS)
    if missing_keys:
        raise ValueError(f"Manifest incomplet pour le run {run_id}: {missing_keys}")

    resolved_source = source_model_path
    if resolved_source is None:
        source_value = manifest.get("source_model_path")
        if not source_value:
            raise ValueError(f"Le manifest du run {run_id} ne contient pas de source_model_path")
        resolved_source = Path(source_value)

    if not resolved_source.exists():
        raise FileNotFoundError(f"Poids source introuvables: {resolved_source}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, target_path)

    manifest["promoted_to_service"] = True
    manifest["promoted_at"] = utc_now_iso()
    manifest["model_name"] = target_path.name
    manifest["model_path"] = str(target_path)

    write_json(manifest_path, manifest)
    write_json(PROMOTED_MANIFEST_PATH, manifest)
    write_json(PROMOTED_METRICS_PATH, metrics)
    write_json(PROMOTED_PARAMS_PATH, params)

    logger.info("Run %s promoted to %s", run_id, target_path)
    return manifest
