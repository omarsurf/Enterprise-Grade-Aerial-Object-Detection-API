"""
Additional tests for FastAPI runtime helpers and lifespan behavior.
"""

import io

import pytest
from fastapi import HTTPException, UploadFile

from app import main


@pytest.mark.asyncio
async def test_read_file_with_limit_rejects_oversized_upload(monkeypatch):
    """Large uploads should fail before image decoding."""
    monkeypatch.setattr(main, "MAX_FILE_SIZE_BYTES", 4)
    monkeypatch.setattr(main, "MAX_FILE_SIZE_MB", 0)
    upload = UploadFile(file=io.BytesIO(b"12345"), filename="big.png")

    with pytest.raises(HTTPException) as exc_info:
        await main.read_file_with_limit(upload)

    assert exc_info.value.status_code == 413


def test_decode_image_rejects_invalid_bytes():
    """Invalid image payloads should raise an HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        main.decode_image(b"not-an-image")

    assert exc_info.value.status_code == 400


def test_build_output_filename_falls_back_to_default_name():
    """Unsafe filenames should be sanitized back to a safe default."""
    generated = main.build_output_filename("../../$$$.png")

    assert generated.startswith("image_")
    assert generated.endswith(".png")


def test_get_runtime_model_info_reflects_detector_state(monkeypatch):
    """Model-info helper should include the detector loaded state."""
    monkeypatch.setattr(main.detector, "is_loaded", lambda: True)
    monkeypatch.setattr(
        main,
        "build_model_info",
        lambda model_loaded: {
            "model_loaded": model_loaded,
            "manifest_available": True,
            "model_name": "best_tiled.pt",
            "model_path": "models/best_tiled.pt",
            "model_version": "run-123",
            "run_id": "run-123",
            "dataset_version": "local-v1",
            "trained_at": "2026-03-24T00:00:00+00:00",
            "metrics_summary": {"mAP50-95": 0.57},
            "message": None,
        },
    )

    payload = main.get_runtime_model_info()

    assert payload.model_loaded is True
    assert payload.run_id == "run-123"


@pytest.mark.asyncio
async def test_lifespan_degrades_on_unexpected_model_load_error(monkeypatch):
    """Unexpected model load failures should keep the API in degraded mode."""

    class DummyExecutor:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self, wait):
            self.shutdown_called = wait

    dummy_executor = DummyExecutor()
    monkeypatch.setattr(main, "_executor", dummy_executor)
    monkeypatch.setattr(main.detector, "load_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(main.detector, "model", object())

    async with main.lifespan(main.app):
        assert main.detector.model is None

    assert dummy_executor.shutdown_called is True
