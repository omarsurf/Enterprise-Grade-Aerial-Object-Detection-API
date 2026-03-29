"""
Integration tests for the prepare_data build and validate workflow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_prepare_data_module():
    """Loads the standalone workflow script as a Python module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("prepare_data_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_source_example(image_path: Path, label_path: Path) -> None:
    """Creates one source image and one DOTA-style label file."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 255, dtype=np.uint8))
    label_path.write_text("8 8 24 8 24 24 8 24 plane 0\n", encoding="utf-8")


def test_prepare_data_builds_tiled_dataset_and_manifest(tmp_path, monkeypatch):
    """Build mode should generate processed tiles, labels, and a dataset manifest."""
    module = load_prepare_data_module()
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts" / "datasets" / "dataset-local-v1"
    train_image = data_root / "filtered" / "split" / "train" / "images" / "train_a.png"
    train_label = data_root / "filtered" / "split" / "train" / "labelTxt" / "train_a.txt"
    val_image = data_root / "filtered" / "split" / "val" / "images" / "val_a.png"
    val_label = data_root / "filtered" / "split" / "val" / "labelTxt" / "val_a.txt"
    config_path = tmp_path / "config.yaml"

    write_source_example(train_image, train_label)
    write_source_example(val_image, val_label)

    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "dataset",
                    "version": "local-v1",
                    "description": "Tiny integration dataset",
                    "raw_dir": str(data_root / "raw"),
                    "filtered_dir": str(data_root / "filtered"),
                    "processed_dir": str(data_root / "processed"),
                    "source_train_images": str(train_image.parent),
                    "source_train_labels": str(train_label.parent),
                    "source_val_images": str(val_image.parent),
                    "source_val_labels": str(val_label.parent),
                    "train_images": str(data_root / "processed" / "split" / "train" / "images"),
                    "train_labels": str(data_root / "processed" / "split" / "train" / "labelTxt"),
                    "val_images": str(data_root / "processed" / "split" / "val" / "images"),
                    "val_labels": str(data_root / "processed" / "split" / "val" / "labelTxt"),
                    "test_images": None,
                    "test_labels": None,
                    "train_split_id": "train-split",
                    "val_split_id": "val-split",
                    "test_split_id": None,
                    "class_names": ["plane"],
                    "tiling": {
                        "tile_size": 32,
                        "overlap": 16,
                        "empty_tile_keep_ratio": 0.0,
                        "random_seed": 7,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "get_dataset_dir", lambda name, version: artifacts_root)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: module.argparse.Namespace(config=str(config_path), mode="build"),
    )

    module.main()

    processed_train_images = sorted((data_root / "processed" / "split" / "train" / "images").glob("*.png"))
    processed_train_labels = sorted(
        (data_root / "processed" / "split" / "train" / "labelTxt").glob("*.txt")
    )
    manifest_path = artifacts_root / "dataset_manifest.json"

    assert processed_train_images
    assert processed_train_labels
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["build_mode"] == "build"
    assert manifest["tiling"]["tile_size"] == 32
    assert manifest["build_stats"]["train"]["tiles_saved"] >= 1
