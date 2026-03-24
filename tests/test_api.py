"""
Tests for the FastAPI endpoints and utility functions.
"""

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.schemas import Detection, ModelInfoResponse, PredictionResponse

# === Fixtures ===


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
    """Create a test client with mocked detector."""
    from app.main import app

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


# === Health Endpoint Tests ===


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data

    def test_health_status_healthy_when_model_loaded(self, client, mock_detector):
        """Status should be 'healthy' when model is loaded."""
        mock_detector.is_loaded.return_value = True
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_status_degraded_when_model_not_loaded(self, client, mock_detector):
        """Status should be 'degraded' when model is not loaded."""
        mock_detector.is_loaded.return_value = False
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""

    def test_model_info_returns_promoted_metadata(self, client):
        """Model-info should expose promoted metadata when available."""
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
            response = client.get("/model-info")

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "best_tiled.pt"
        assert data["run_id"] == "run-123"
        assert data["manifest_available"] is True

    def test_model_info_signals_missing_manifest(self, client):
        """Model-info should signal when the promoted manifest is missing."""
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
            response = client.get("/model-info")

        assert response.status_code == 200
        data = response.json()
        assert data["manifest_available"] is False
        assert "manifest" in data["message"].lower()


# === Predict Endpoint Tests ===


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_requires_file(self, client):
        """Predict should return 422 without file."""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_invalid_file_type(self, client):
        """Predict should return 400 for invalid file type."""
        response = client.post(
            "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "Extension" in response.json()["error"]

    def test_predict_valid_png_image(self, client, sample_image_bytes, mock_detector):
        """Predict should succeed with valid PNG image."""
        response = client.post(
            "/predict", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["image_width"] == 512
        assert data["image_height"] == 512
        assert data["nb_detections"] == 2
        assert len(data["detections"]) == 2

    def test_predict_valid_jpeg_image(self, client, sample_jpeg_bytes, mock_detector):
        """Predict should succeed with valid JPEG image."""
        response = client.post(
            "/predict", files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_detection_structure(self, client, sample_image_bytes, mock_detector):
        """Detections should have correct structure."""
        response = client.post(
            "/predict", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        data = response.json()
        detection = data["detections"][0]

        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "polygon" in detection
        assert len(detection["polygon"]) == 4

    def test_predict_returns_503_when_model_not_loaded(
        self, client, mock_detector, sample_image_bytes
    ):
        """Predict should return 503 when model is not loaded."""
        mock_detector.is_loaded.return_value = False
        response = client.post(
            "/predict", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 503


# === Predict and Save Endpoint Tests ===


class TestPredictAndSaveEndpoint:
    """Tests for /predict-and-save endpoint."""

    def test_predict_and_save_requires_file(self, client):
        """Predict-and-save should return 422 without file."""
        response = client.post("/predict-and-save")
        assert response.status_code == 422

    def test_predict_and_save_valid_image(self, client, sample_image_bytes, mock_detector):
        """Predict-and-save should succeed with valid image."""
        response = client.post(
            "/predict-and-save", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 200

        data = response.json()
        assert "output_path" in data
        assert data["nb_detections"] == 2

    def test_predict_and_save_returns_503_when_model_not_loaded(
        self, client, mock_detector, sample_image_bytes
    ):
        """Predict-and-save should return 503 when model is not loaded."""
        mock_detector.is_loaded.return_value = False
        response = client.post(
            "/predict-and-save", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 503

    def test_predict_and_save_sanitizes_output_filename(
        self, client, sample_image_bytes, mock_detector
    ):
        """Uploaded filename should not escape the predictions directory."""
        response = client.post(
            "/predict-and-save",
            files={"file": ("../../escape.png", sample_image_bytes, "image/png")},
        )

        assert response.status_code == 200
        generated_filename = mock_detector.predict_and_save.call_args[0][1]
        assert generated_filename.startswith("escape_")
        assert ".." not in generated_filename
        assert "/" not in generated_filename
        assert "\\" not in generated_filename
        assert generated_filename.endswith(".png")


# === Utility Functions Tests ===


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_tiles_coverage(self):
        """Tiles should cover the entire image."""
        from app.utils import generate_tiles

        width, height = 2048, 2048
        tile_size, overlap = 1024, 200

        tiles = list(generate_tiles(width, height, tile_size, overlap))

        # Should have multiple tiles
        assert len(tiles) > 1

        # Each tile should be valid
        for x_start, y_start, x_end, y_end in tiles:
            assert 0 <= x_start < x_end <= width
            assert 0 <= y_start < y_end <= height

    def test_generate_tiles_small_image(self):
        """Small image should generate single tile."""
        from app.utils import generate_tiles

        tiles = list(generate_tiles(512, 512, 1024, 200))

        # Small image = single tile
        assert len(tiles) == 1
        x_start, y_start, x_end, y_end = tiles[0]
        assert x_start == 0 and y_start == 0
        assert x_end == 512 and y_end == 512

    def test_generate_tiles_distributes_overlap_evenly(self):
        """Large images should avoid oversized last-tile overlap."""
        from app.utils import generate_tiles

        tiles = list(generate_tiles(2048, 2048, 1024, 200))
        x_tiles = [tile for tile in tiles if tile[1] == 0]

        assert x_tiles == [
            (0, 0, 1024, 1024),
            (512, 0, 1536, 1024),
            (1024, 0, 2048, 1024),
        ]

    def test_generate_tiles_does_not_duplicate_last_tile(self):
        """Edge coverage should not emit the same tile twice."""
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
        """IoU of identical boxes should be 1."""
        from app.utils import compute_iou_aabb

        box = (0, 0, 100, 100)
        assert compute_iou_aabb(box, box) == 1.0

    def test_compute_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0."""
        from app.utils import compute_iou_aabb

        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou_aabb(box1, box2) == 0.0

    def test_compute_iou_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        from app.utils import compute_iou_aabb

        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = compute_iou_aabb(box1, box2)
        assert 0 < iou < 1

    def test_obb_to_aabb(self):
        """OBB to AABB conversion should work correctly."""
        from app.utils import obb_to_aabb

        obb = np.array([[10, 20], [50, 25], [45, 60], [5, 55]])

        x_min, y_min, x_max, y_max = obb_to_aabb(obb)

        assert x_min == 5
        assert y_min == 20
        assert x_max == 50
        assert y_max == 60

    def test_reproject_obb_to_global(self):
        """Reprojection should add offset correctly."""
        from app.utils import reproject_obb_to_global

        local_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        offset_x, offset_y = 100, 200

        global_points = reproject_obb_to_global(local_points, offset_x, offset_y)

        expected = np.array([[100, 200], [110, 200], [110, 210], [100, 210]])
        np.testing.assert_array_equal(global_points, expected)

    def test_nms_obb_removes_duplicates(self):
        """NMS should remove overlapping detections of same class."""
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

        # Should keep only the higher confidence detection
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_nms_obb_keeps_different_classes(self):
        """NMS should keep overlapping detections of different classes."""
        from app.utils import nms_obb

        detections = [
            {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "confidence": 0.9, "class_id": 0},
            {
                "polygon": [[10, 10], [110, 10], [110, 110], [10, 110]],
                "confidence": 0.8,
                "class_id": 1,
            },
        ]

        result = nms_obb(detections, iou_threshold=0.5)

        # Should keep both (different classes)
        assert len(result) == 2


# === Schema Validation Tests ===


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_detection_valid(self):
        """Valid Detection should be created without errors."""
        detection = Detection(
            class_id=0,
            class_name="plane",
            confidence=0.95,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        assert detection.class_id == 0
        assert detection.confidence == 0.95

    def test_detection_invalid_polygon_too_few_points(self):
        """Detection with too few polygon points should raise error."""
        with pytest.raises(ValueError, match="4 points"):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=0.95,
                polygon=[[0, 0], [10, 0], [10, 10]],  # Only 3 points
            )

    def test_detection_invalid_polygon_too_many_points(self):
        """Detection with too many polygon points should raise error."""
        with pytest.raises(ValueError, match="4 points"):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=0.95,
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],  # 5 points
            )

    def test_detection_invalid_confidence(self):
        """Detection with invalid confidence should raise error."""
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=1.5,  # > 1
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            )

    def test_detection_negative_confidence(self):
        """Detection with negative confidence should raise error."""
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="plane",
                confidence=-0.5,  # < 0
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            )
