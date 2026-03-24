"""
Tests for lightweight artifact helpers used by the local MLOps workflow.
"""

from app import artifacts


def test_build_model_info_without_promoted_manifest(monkeypatch, tmp_path):
    """Missing promoted metadata should be signaled explicitly."""
    monkeypatch.setattr(artifacts, "PROMOTED_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(artifacts, "PROMOTED_METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(artifacts, "PROMOTED_PARAMS_PATH", tmp_path / "params.json")
    monkeypatch.setattr(artifacts, "MODEL_PATH", tmp_path / "best_tiled.pt")

    payload = artifacts.build_model_info(model_loaded=False)

    assert payload["manifest_available"] is False
    assert payload["model_name"] == "best_tiled.pt"
    assert "manifest" in payload["message"].lower()


def test_build_model_info_with_promoted_manifest(monkeypatch, tmp_path):
    """Promoted metadata should be loaded into the runtime payload."""
    promoted_dir = tmp_path / "promoted"
    promoted_dir.mkdir(parents=True)

    manifest_path = promoted_dir / "manifest.json"
    metrics_path = promoted_dir / "metrics.json"
    params_path = promoted_dir / "params.json"
    model_path = tmp_path / "best_tiled.pt"

    manifest_path.write_text(
        """
{
  "model_name": "best_tiled.pt",
  "model_version": "run-001",
  "run_id": "run-001",
  "dataset_version": "local-v1",
  "training_date": "2026-03-23T00:00:00+00:00",
  "metrics_summary": {
    "mAP50-95": 0.57
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text('{"metrics_summary": {"mAP50-95": 0.57}}\n', encoding="utf-8")
    params_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(artifacts, "PROMOTED_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(artifacts, "PROMOTED_METRICS_PATH", metrics_path)
    monkeypatch.setattr(artifacts, "PROMOTED_PARAMS_PATH", params_path)
    monkeypatch.setattr(artifacts, "MODEL_PATH", model_path)

    payload = artifacts.build_model_info(model_loaded=True)

    assert payload["manifest_available"] is True
    assert payload["model_version"] == "run-001"
    assert payload["metrics_summary"]["mAP50-95"] == 0.57
    assert payload["message"] is None


def test_validate_required_keys_reports_missing_entries():
    """Manifest validation should report missing required keys clearly."""
    payload = {"run_id": "run-123", "model_name": "best_tiled.pt"}

    missing = artifacts.validate_required_keys(payload, artifacts.REQUIRED_MODEL_MANIFEST_KEYS)

    assert "dataset_name" in missing
    assert "metrics_summary" in missing
