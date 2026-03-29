"""
Tests for the FastAPI endpoints, runtime controls, and utility helpers.
"""

import asyncio
import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import API_KEY_HEADER_NAME, DEFAULT_DEV_API_KEY
from app.schemas import Detection, ModelInfoResponse, PredictionResponse

AUTH_HEADERS = {API_KEY_HEADER_NAME: DEFAULT_DEV_API_KEY}


@pytest.fixture
def mock_detector():
    """Mock the detector to avoid loading the real YOLO model."""
    with patch("app.main.detector") as mock:
        mock.is_loaded.return_value = True
        mock.predict.return_value = PredictionResponse(
            image_width=512,
            image_height=512,
            nb_detections=2,
            detections=[
                Detection(
                    class_id=0,
                    class_name="plane",
                    confidence=0.95,
                    polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
                ),
                Detection(
                    class_id=1,
                    class_name="ship",
                    confidence=0.87,
                    polygon=[[300, 300], [400, 300], [400, 400], [300, 400]],
                ),
            ],
        )
        mock.predict_and_save.return_value = (
            mock.predict.return_value,
            "/app/outputs/predictions/test_image.png",
        )
        yield mock


@pytest.fixture
def client(mock_detector):
    """Create a test client with mocked detector and reset rate-limit state."""
    from app.main import app, limiter

    storage = getattr(limiter, "_storage", None)
    for method_name in ("reset", "clear"):
        method = getattr(storage, method_name, None)
        if callable(method):
            method()
            break

    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample PNG image for testing."""
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes():
    """Create a sample JPEG image for testing."""
    img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200_without_auth(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data
        assert "x-request-id" in {key.lower() for key in response.headers}

    def test_health_status_degraded_when_model_not_loaded(self, client, mock_detector):
        mock_detector.is_loaded.return_value = False
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""

    def test_model_info_requires_api_key(self, client):
        response = client.get("/model-info")
        assert response.status_code == 401

    def test_model_info_rejects_invalid_api_key(self, client):
        response = client.get("/model-info", headers={API_KEY_HEADER_NAME: "wrong-key"})
        assert response.status_code == 401

    def test_model_info_returns_promoted_metadata(self, client):
        mocked_payload = ModelInfoResponse(
            model_loaded=True,
            manifest_available=True,
            model_name="best_tiled.pt",
            model_path="models/best_tiled.pt",
            model_version="run-123",
            run_id="run-123",
            dataset_version="local-v1",
            trained_at="2026-03-23T00:00:00+00:00",
            metrics_summary={"mAP50-95": 0.57},
            message=None,
        )

        with patch("app.main.get_runtime_model_info", return_value=mocked_payload):
            response = client.get("/model-info", headers=AUTH_HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "best_tiled.pt"
        assert data["run_id"] == "run-123"
        assert data["manifest_available"] is True

    def test_model_info_signals_missing_manifest(self, client):
        mocked_payload = ModelInfoResponse(
            model_loaded=False,
            manifest_available=False,
            model_name="best_tiled.pt",
            model_path="models/best_tiled.pt",
            model_version=None,
            run_id=None,
            dataset_version=None,
            trained_at=None,
            metrics_summary={},
            message="Promoted model manifest not found.",
        )

        with patch("app.main.get_runtime_model_info", return_value=mocked_payload):
            response = client.get("/model-info", headers=AUTH_HEADERS)

        assert response.status_code == 200
        assert response.json()["manifest_available"] is False

    def test_model_info_rate_limited(self, client):
        for _ in range(30):
            response = client.get("/model-info", headers=AUTH_HEADERS)
            assert response.status_code == 200

        response = client.get("/model-info", headers=AUTH_HEADERS)
        assert response.status_code == 429


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_requires_api_key(self, client, sample_image_bytes):
        response = client.post(
            "/predict", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 401

    def test_predict_requires_file_when_authenticated(self, client):
        response = client.post("/predict", headers=AUTH_HEADERS)
        assert response.status_code == 422

    def test_predict_invalid_file_type(self, client):
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Extension" in response.json()["error"]

    def test_predict_valid_png_image(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["image_width"] == 512
        assert data["image_height"] == 512
        assert data["nb_detections"] == 2
        assert len(data["detections"]) == 2

    def test_predict_valid_jpeg_image(self, client, sample_jpeg_bytes):
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_detection_structure(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        detection = response.json()["detections"][0]

        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "polygon" in detection
        assert len(detection["polygon"]) == 4

    def test_predict_returns_503_when_model_not_loaded(self, client, mock_detector, sample_image_bytes):
        mock_detector.is_loaded.return_value = False
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 503

    def test_predict_echoes_request_id(self, client, sample_image_bytes):
        response = client.post(
            "/predict",
            headers={**AUTH_HEADERS, "X-Request-ID": "req-123"},
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "req-123"

    def test_predict_returns_503_when_inference_slots_are_saturated(
        self, client, sample_image_bytes, monkeypatch
    ):
        from app import main

        monkeypatch.setattr(main, "_prediction_semaphore", asyncio.Semaphore(0))
        response = client.post(
            "/predict",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == str(main.BUSY_RETRY_AFTER_SECONDS)


class TestPredictAndSaveEndpoint:
    """Tests for /predict-and-save endpoint."""

    def test_predict_and_save_requires_api_key(self, client, sample_image_bytes):
        response = client.post(
            "/predict-and-save", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 401

    def test_predict_and_save_requires_file_when_authenticated(self, client):
        response = client.post("/predict-and-save", headers=AUTH_HEADERS)
        assert response.status_code == 422

    def test_predict_and_save_valid_image(self, client, sample_image_bytes):
        response = client.post(
            "/predict-and-save",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "output_path" in data
        assert data["nb_detections"] == 2

    def test_predict_and_save_returns_503_when_model_not_loaded(
        self, client, mock_detector, sample_image_bytes
    ):
        mock_detector.is_loaded.return_value = False
        response = client.post(
            "/predict-and-save",
            headers=AUTH_HEADERS,
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 503

    def test_predict_and_save_sanitizes_output_filename(
        self, client, sample_image_bytes, mock_detector
    ):
        response = client.post(
            "/predict-and-save",
            headers=AUTH_HEADERS,
            files={"file": ("../../escape.png", sample_image_bytes, "image/png")},
        )

        assert response.status_code == 200
        generated_filename = mock_detector.predict_and_save.call_args[0][1]
        assert generated_filename.startswith("escape_")
        assert ".." not in generated_filename
        assert "/" not in generated_filename
        assert "\\" not in generated_filename
        assert generated_filename.endswith(".png")


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_tiles_coverage(self):
        from app.utils import generate_tiles

        tiles = list(generate_tiles(2048, 2048, 1024, 200))

        assert len(tiles) > 1
        for x_start, y_start, x_end, y_end in tiles:
            assert 0 <= x_start < x_end <= 2048
            assert 0 <= y_start < y_end <= 2048

    def test_generate_tiles_small_image(self):
        from app.utils import generate_tiles

        tiles = list(generate_tiles(512, 512, 1024, 200))
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 512, 512)

    def test_generate_tiles_distributes_overlap_evenly(self):
        from app.utils import generate_tiles

        tiles = list(generate_tiles(2048, 2048, 1024, 200))
        x_tiles = [tile for tile in tiles if tile[1] == 0]

        assert x_tiles == [
            (0, 0, 1024, 1024),
            (512, 0, 1536, 1024),
            (1024, 0, 2048, 1024),
        ]

    def test_generate_tiles_does_not_duplicate_last_tile(self):
        from app.utils import generate_tiles

        tiles = list(generate_tiles(2500, 2500, 1024, 200))
        x_tiles = [tile for tile in tiles if tile[1] == 0]

        assert len(x_tiles) == len(set(x_tiles))
        assert x_tiles == [
            (0, 0, 1024, 1024),
            (738, 0, 1762, 1024),
            (1476, 0, 2500, 1024),
        ]

    def test_compute_iou_identical_boxes(self):
        from app.utils import compute_iou_aabb

        box = (0, 0, 100, 100)
        assert compute_iou_aabb(box, box) == 1.0

    def test_compute_iou_no_overlap(self):
        from app.utils import compute_iou_aabb

        assert compute_iou_aabb((0, 0, 50, 50), (100, 100, 150, 150)) == 0.0

    def test_compute_iou_partial_overlap(self):
        from app.utils import compute_iou_aabb

        iou = compute_iou_aabb((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0 < iou < 1

    def test_obb_to_aabb(self):
        from app.utils import obb_to_aabb

        obb = np.array([[10, 20], [50, 25], [45, 60], [5, 55]])
        assert obb_to_aabb(obb) == (5, 20, 50, 60)

    def test_reproject_obb_to_global(self):
        from app.utils import reproject_obb_to_global

        local_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        global_points = reproject_obb_to_global(local_points, 100, 200)
        expected = np.array([[100, 200], [110, 200], [110, 210], [100, 210]])
        np.testing.assert_array_equal(global_points, expected)

    def test_nms_obb_removes_duplicates(self):
        from app.utils import nms_obb

        detections = [
            {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "confidence": 0.9, "class_id": 0},
            {
                "polygon": [[10, 10], [110, 10], [110, 110], [10, 110]],
                "confidence": 0.8,
                "class_id": 0,
            },
        ]

        result = nms_obb(detections, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_nms_obb_keeps_different_classes(self):
        from app.utils import nms_obb

        detections = [
            {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "confidence": 0.9, "class_id": 0},
            {
                "polygon": [[10, 10], [110, 10], [110, 110], [10, 110]],
                "confidence": 0.8,
                "class_id": 1,
            },
        ]

        assert len(nms_obb(detections, iou_threshold=0.5)) == 2


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_detection_valid(self):
        detection = Detection(
            class_id=0,
            class_name="plane",
            confidence=0.95,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        assert detection.class_id == 0
        assert detection.confidence == 0.95

    def test_detection_invalid_polygon_too_few_points(self):
        with pytest.raises(ValueError, match="4 points"):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=0.95,
                polygon=[[0, 0], [10, 0], [10, 10]],
            )

    def test_detection_invalid_polygon_too_many_points(self):
        with pytest.raises(ValueError, match="4 points"):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=0.95,
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],
            )

    def test_detection_invalid_confidence(self):
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=1.5,
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            )

    def test_detection_negative_confidence(self):
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=-0.5,
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            )
