"""
Focused tests for inference, tiling utilities, and annotated output generation.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import inference
from app.utils import (
    compute_iou_aabb,
    draw_obb_on_image,
    extract_tile,
    generate_tile_grid,
    is_detection_in_padding,
)


class FakeTensor:
    """Mimics the minimal tensor API used by Ultralytics results."""

    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)


class FakeBoxes:
    """Simple container exposing the Ultralytics OBB attributes used by the detector."""

    def __init__(self):
        self.xyxyxyxy = [
            FakeTensor([[0, 0], [10, 0], [10, 5], [0, 5]]),
            FakeTensor([[2, 2], [6, 2], [6, 6], [2, 6]]),
        ]
        self.conf = [FakeTensor(0.91), FakeTensor(0.42)]
        self.cls = [FakeTensor(2), FakeTensor(1)]

    def __len__(self):
        return len(self.xyxyxyxy)


def test_load_model_raises_when_weights_are_missing(tmp_path):
    """Loading should fail fast when the configured weights file is absent."""
    detector = inference.OBBDetector(tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError):
        detector.load_model()


def test_load_model_initializes_yolo(monkeypatch, tmp_path):
    """Loading should instantiate the YOLO wrapper with the resolved path."""
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    created = {}

    class DummyYOLO:
        def __init__(self, path):
            created["path"] = path

    monkeypatch.setattr(inference, "YOLO", DummyYOLO)
    detector = inference.OBBDetector(model_path)

    detector.load_model()

    assert created["path"] == str(model_path)
    assert isinstance(detector.model, DummyYOLO)


def test_predict_tile_formats_yolo_obb_output():
    """Tile inference should extract polygons, confidence, and classes."""
    detector = inference.OBBDetector()
    boxes = FakeBoxes()
    detector.model = SimpleNamespace(predict=lambda *args, **kwargs: [SimpleNamespace(obb=boxes)])

    detections = detector.predict_tile(np.zeros((16, 16, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0]["class_id"] == 2
    assert detections[0]["confidence"] == pytest.approx(0.91)
    assert detections[0]["polygon"].shape == (4, 2)


def test_predict_full_image_filters_padding_and_clears_gpu(monkeypatch):
    """Global inference should reproject detections, filter padding, and release resources."""
    detector = inference.OBBDetector()
    detector.model = object()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    clear_calls = []

    monkeypatch.setattr(
        inference,
        "generate_tile_grid",
        lambda *args, **kwargs: iter(
            [
                ((0, 0, 8, 8), (0.0, 0.0, 8.0, 8.0)),
                ((8, 0, 16, 8), (8.0, 0.0, 16.0, 8.0)),
            ]
        ),
    )
    monkeypatch.setattr(
        inference, "extract_tile", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(detector, "_clear_gpu_memory", lambda: clear_calls.append(True))
    monkeypatch.setattr(
        detector,
        "predict_tile",
        lambda tile: [{"polygon": np.array([[0, 0], [2, 0], [2, 2], [0, 2]]), "confidence": 0.9, "class_id": 0}],
    )
    monkeypatch.setattr(
        inference,
        "is_detection_in_padding",
        lambda polygon, valid_region: valid_region[0] >= 8.0,
    )
    monkeypatch.setattr(inference, "nms_obb", lambda detections, threshold: detections)

    detections, width, height = detector.predict_full_image(image)

    assert width == 16
    assert height == 12
    assert len(detections) == 1
    assert detections[0]["polygon"][0] == [0, 0]
    assert clear_calls == [True]


def test_predict_builds_response_from_raw_detections(monkeypatch):
    """High-level predict should format class names and rounded confidences."""
    detector = inference.OBBDetector()
    detector.model = object()
    image = np.zeros((20, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(
        detector,
        "predict_full_image",
        lambda _: (
            [{"polygon": [[1, 1], [5, 1], [5, 4], [1, 4]], "confidence": 0.91234, "class_id": 0}],
            10,
            20,
        ),
    )

    response = detector.predict(image)

    assert response.image_width == 10
    assert response.nb_detections == 1
    assert response.detections[0].class_name == "plane"
    assert response.detections[0].confidence == 0.9123


def test_predict_and_save_writes_annotated_image(monkeypatch, tmp_path):
    """Saving predictions should write an annotated image inside the outputs directory."""
    detector = inference.OBBDetector()
    detector.model = object()
    image = np.zeros((24, 24, 3), dtype=np.uint8)

    monkeypatch.setattr(
        detector,
        "predict_full_image",
        lambda _: (
            [{"polygon": [[2, 2], [10, 2], [10, 10], [2, 10]], "confidence": 0.95, "class_id": 0}],
            24,
            24,
        ),
    )
    monkeypatch.setattr(inference, "OUTPUTS_DIR", tmp_path)

    response, output_path = detector.predict_and_save(image, "annotated.png")

    assert response.nb_detections == 1
    assert Path(output_path).exists()


def test_predict_and_save_rejects_path_escape(monkeypatch, tmp_path):
    """Output filenames must stay inside the configured predictions directory."""
    detector = inference.OBBDetector()
    detector.model = object()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(detector, "predict_full_image", lambda _: ([], 8, 8))
    monkeypatch.setattr(inference, "OUTPUTS_DIR", tmp_path)

    with pytest.raises(ValueError):
        detector.predict_and_save(image, "../escape.png")


def test_generate_tile_grid_and_padding_logic():
    """Tile grids should expose the valid overlap-splitting regions."""
    first_row = [tile for tile in generate_tile_grid(2048, 1024, 1024, 200) if tile[0][1] == 0]

    assert first_row[0][1] == (0.0, 0.0, 768.0, 1024.0)
    assert first_row[1][1] == (768.0, 0.0, 1280.0, 1024.0)
    assert first_row[2][1] == (1280.0, 0.0, 2048.0, 1024.0)

    inside = np.array([[100, 100], [140, 100], [140, 140], [100, 140]])
    outside = np.array([[10, 100], [40, 100], [40, 140], [10, 140]])

    assert is_detection_in_padding(
        inside, tile_coords=(50, 50, 250, 250), image_width=300, image_height=300, overlap=100
    ) is False
    assert is_detection_in_padding(
        outside, tile_coords=(50, 50, 250, 250), image_width=300, image_height=300, overlap=100
    ) is True

    with pytest.raises(ValueError):
        is_detection_in_padding(inside)


def test_draw_obb_extract_tile_and_zero_union_iou():
    """Utility helpers should annotate images, return tile copies, and handle degenerate IoU."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    tile = extract_tile(image, (0, 0, 10, 10))
    tile[:, :] = 255

    assert np.count_nonzero(image) == 0

    annotated = draw_obb_on_image(
        image,
        [{"polygon": [[2, 2], [8, 2], [8, 8], [2, 8]], "class_id": 0, "class_name": "plane", "confidence": 0.9}],
        {0: (0, 255, 0)},
    )

    assert np.count_nonzero(annotated) > 0
    assert compute_iou_aabb((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
