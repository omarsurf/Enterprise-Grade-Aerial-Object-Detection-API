"""
Tests for the dataset tiling helpers extracted from notebooks.
"""

from pathlib import Path

import cv2
import numpy as np

from app.dataset_tiling import (
    SourceObject,
    TileWindow,
    build_tile_annotations,
    deterministic_keep_empty_tile,
    generate_tile_windows,
    list_image_paths,
    read_dota_label,
)


def test_read_dota_label_filters_metadata_and_unknown_classes(tmp_path: Path):
    """Only valid DOTA annotation lines for known classes should be returned."""
    label_path = tmp_path / "sample.txt"
    label_path.write_text(
        "\n".join(
            [
                "imagesource:GoogleEarth",
                "gsd:0.12",
                "0 0 10 0 10 10 0 10 plane 0",
                "5 5 15 5 15 15 5 15 harbor 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    objects = read_dota_label(label_path, {"plane": 0})

    assert len(objects) == 1
    assert objects[0].class_id == 0
    assert objects[0].class_name == "plane"
    assert objects[0].polygon[2] == (10.0, 10.0)


def test_generate_tile_windows_uses_notebook_stride():
    """Training tiles should follow the original notebook stride pattern."""
    windows = generate_tile_windows(1500, 1100, tile_size=1024, overlap=200)

    assert windows[0] == TileWindow(index=0, x=0, y=0, width=1024, height=1024)
    assert windows[1] == TileWindow(index=1, x=824, y=0, width=676, height=1024)
    assert windows[-1] == TileWindow(index=3, x=824, y=824, width=676, height=276)


def test_build_tile_annotations_clips_partial_objects():
    """Partially visible OBBs should be clipped and normalized into YOLO OBB format."""
    source_object = SourceObject(
        class_id=0,
        class_name="plane",
        polygon=((8, 8), (24, 8), (24, 24), (8, 24)),
    )
    window = TileWindow(index=0, x=0, y=0, width=16, height=16)

    annotations = build_tile_annotations([source_object], window, tile_size=32)

    assert len(annotations) == 1
    class_id, coords = annotations[0]
    assert class_id == 0
    assert len(coords) == 8
    assert all(0.0 <= coord <= 1.0 for coord in coords)


def test_deterministic_keep_empty_tile_is_stable():
    """Empty-tile sampling should be deterministic for a given seed."""
    first = deterministic_keep_empty_tile("image-1", 5, 0.25, 42)
    second = deterministic_keep_empty_tile("image-1", 5, 0.25, 42)
    third = deterministic_keep_empty_tile("image-1", 6, 0.25, 42)

    assert first == second
    assert first in {True, False}
    assert third in {True, False}


def test_list_image_paths_filters_supported_extensions(tmp_path: Path):
    """Only supported image files should be returned."""
    image_path = tmp_path / "sample.png"
    other_path = tmp_path / "notes.txt"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    other_path.write_text("not an image\n", encoding="utf-8")

    assert list_image_paths(tmp_path) == [image_path]
