#!/usr/bin/env python3
"""
Validates the local dataset layout and writes a reproducible dataset manifest.
"""

import argparse
from pathlib import Path

from common import load_yaml_config, resolve_repo_path, write_yaml_config

from app.artifacts import (  # noqa: E402
    REQUIRED_DATASET_MANIFEST_KEYS,
    get_dataset_dir,
    utc_now_iso,
    validate_required_keys,
    write_json,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset paths and emit a dataset manifest."
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to the dataset config YAML file.",
    )
    return parser.parse_args()


def list_stems(directory: Path, suffixes: tuple[str, ...]) -> list[str]:
    return sorted(path.stem for path in directory.iterdir() if path.suffix.lower() in suffixes)


def ensure_label_alias(split_dir: Path, label_dir: Path) -> dict[str, str]:
    """Creates a labels symlink next to YOLO image folders when possible."""
    alias_dir = split_dir / "labels"

    if alias_dir.exists():
        return {"labels_alias": str(alias_dir), "labels_alias_status": "existing"}

    try:
        alias_dir.symlink_to(label_dir.resolve(), target_is_directory=True)
        return {"labels_alias": str(alias_dir), "labels_alias_status": "created"}
    except OSError:
        return {"labels_alias": str(label_dir), "labels_alias_status": "unavailable"}


def validate_split(images_dir: Path, labels_dir: Path, split_name: str) -> dict[str, object]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory missing for {split_name}: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory missing for {split_name}: {labels_dir}")

    image_stems = list_stems(images_dir, tuple(IMAGE_EXTENSIONS))
    label_stems = list_stems(labels_dir, (".txt",))

    if not image_stems:
        raise ValueError(f"No images found for {split_name}: {images_dir}")
    if not label_stems:
        raise ValueError(f"No label files found for {split_name}: {labels_dir}")

    missing_labels = sorted(set(image_stems) - set(label_stems))
    extra_labels = sorted(set(label_stems) - set(image_stems))
    if missing_labels:
        raise ValueError(f"Missing labels for {split_name}: {missing_labels[:10]}")
    if extra_labels:
        raise ValueError(f"Orphan labels for {split_name}: {extra_labels[:10]}")

    alias_info = ensure_label_alias(images_dir.parent, labels_dir)
    return {
        "split_name": split_name,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "num_images": len(image_stems),
        "num_labels": len(label_stems),
        **alias_info,
    }


def build_ultralytics_data_file(dataset_cfg: dict[str, object], dataset_dir: Path) -> Path:
    processed_root = resolve_repo_path(str(dataset_cfg["processed_dir"]))
    train_images = resolve_repo_path(str(dataset_cfg["train_images"]))
    val_images = resolve_repo_path(str(dataset_cfg["val_images"]))

    payload = {
        "path": str(processed_root),
        "train": str(train_images.relative_to(processed_root)),
        "val": str(val_images.relative_to(processed_root)),
        "names": {index: name for index, name in enumerate(dataset_cfg["class_names"])},
    }

    ultralytics_data_file = dataset_dir / "ultralytics_data.yaml"
    write_yaml_config(ultralytics_data_file, payload)
    return ultralytics_data_file


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    dataset_cfg = config["dataset"]

    dataset_dir = get_dataset_dir(
        str(dataset_cfg["name"]),
        str(dataset_cfg["version"]),
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_validation = validate_split(
        resolve_repo_path(str(dataset_cfg["train_images"])),
        resolve_repo_path(str(dataset_cfg["train_labels"])),
        "train",
    )
    val_validation = validate_split(
        resolve_repo_path(str(dataset_cfg["val_images"])),
        resolve_repo_path(str(dataset_cfg["val_labels"])),
        "val",
    )

    ultralytics_data_file = build_ultralytics_data_file(dataset_cfg, dataset_dir)
    dataset_manifest = {
        "dataset_name": dataset_cfg["name"],
        "dataset_version": dataset_cfg["version"],
        "description": dataset_cfg["description"],
        "dataset_root": str(resolve_repo_path(str(dataset_cfg["processed_dir"]))),
        "raw_dir": str(resolve_repo_path(str(dataset_cfg["raw_dir"]))),
        "filtered_dir": str(resolve_repo_path(str(dataset_cfg["filtered_dir"]))),
        "processed_dir": str(resolve_repo_path(str(dataset_cfg["processed_dir"]))),
        "train_images": train_validation["images_dir"],
        "train_labels": train_validation["labels_dir"],
        "val_images": val_validation["images_dir"],
        "val_labels": val_validation["labels_dir"],
        "class_names": dataset_cfg["class_names"],
        "train_split_id": dataset_cfg["train_split_id"],
        "val_split_id": dataset_cfg["val_split_id"],
        "test_split_id": dataset_cfg.get("test_split_id"),
        "num_train_images": train_validation["num_images"],
        "num_val_images": val_validation["num_images"],
        "ultralytics_data_file": str(ultralytics_data_file),
        "generated_at": utc_now_iso(),
        "validation": {
            "train": train_validation,
            "val": val_validation,
        },
    }

    missing_keys = validate_required_keys(dataset_manifest, REQUIRED_DATASET_MANIFEST_KEYS)
    if missing_keys:
        raise ValueError(f"Dataset manifest missing required keys: {missing_keys}")

    manifest_path = dataset_dir / "dataset_manifest.json"
    write_json(manifest_path, dataset_manifest)

    print(f"[OK] Dataset manifest written to {manifest_path}")
    print(f"[OK] Ultralytics data file written to {ultralytics_data_file}")


if __name__ == "__main__":
    main()
