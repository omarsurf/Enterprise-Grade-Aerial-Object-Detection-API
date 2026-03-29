"""
Helpers for building a tiled YOLO OBB dataset from DOTA-style source labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceObject:
    """A single DOTA-style object annotation in absolute image coordinates."""

    class_id: int
    class_name: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TileWindow:
    """A tiled window over the source image."""

    index: int
    x: int
    y: int
    width: int
    height: int


def list_image_paths(directory: Path) -> list[Path]:
    """Returns sorted image files from a directory."""
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_dota_label(
    file_path: Path,
    class_to_id: dict[str, int] | None = None,
) -> list[SourceObject]:
    """
    Reads a DOTA-style annotation file.

    Supported lines start with 8 polygon coordinates followed by a class label.
    Extra trailing tokens such as difficulty are ignored.
    """
    objects: list[SourceObject] = []

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 9:
            continue

        try:
            coords = [float(value) for value in parts[:8]]
        except ValueError:
            continue

        class_name = parts[8]
        if class_to_id is not None and class_name not in class_to_id:
            continue

        class_id = class_to_id[class_name] if class_to_id is not None else -1
        polygon = tuple((coords[index], coords[index + 1]) for index in range(0, 8, 2))
        objects.append(
            SourceObject(
                class_id=class_id,
                class_name=class_name,
                polygon=polygon,
            )
        )

    return objects


def generate_tile_windows(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> list[TileWindow]:
    """Builds deterministic windows using the original notebook tiling stride."""
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    step = tile_size - overlap
    windows: list[TileWindow] = []
    index = 0

    for y in range(0, image_height, step):
        for x in range(0, image_width, step):
            width = min(tile_size, image_width - x)
            height = min(tile_size, image_height - y)
            windows.append(TileWindow(index=index, x=x, y=y, width=width, height=height))
            index += 1

    return windows


def extract_padded_tile(image: np.ndarray, window: TileWindow, tile_size: int) -> np.ndarray:
    """Extracts a tile and pads it to a fixed square size."""
    tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
    crop = image[window.y : window.y + window.height, window.x : window.x + window.width]
    tile[: window.height, : window.width] = crop
    return tile


def _clip_against_vertical(points: list[np.ndarray], x_limit: float, keep_greater: bool) -> list[np.ndarray]:
    if not points:
        return []

    clipped: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = previous[0] >= x_limit if keep_greater else previous[0] <= x_limit

    for current in points:
        current_inside = current[0] >= x_limit if keep_greater else current[0] <= x_limit

        if current_inside != previous_inside and current[0] != previous[0]:
            ratio = (x_limit - previous[0]) / (current[0] - previous[0])
            y_coord = previous[1] + ratio * (current[1] - previous[1])
            clipped.append(np.array([x_limit, y_coord], dtype=np.float32))

        if current_inside:
            clipped.append(current)

        previous = current
        previous_inside = current_inside

    return clipped


def _clip_against_horizontal(points: list[np.ndarray], y_limit: float, keep_greater: bool) -> list[np.ndarray]:
    if not points:
        return []

    clipped: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = previous[1] >= y_limit if keep_greater else previous[1] <= y_limit

    for current in points:
        current_inside = current[1] >= y_limit if keep_greater else current[1] <= y_limit

        if current_inside != previous_inside and current[1] != previous[1]:
            ratio = (y_limit - previous[1]) / (current[1] - previous[1])
            x_coord = previous[0] + ratio * (current[0] - previous[0])
            clipped.append(np.array([x_coord, y_limit], dtype=np.float32))

        if current_inside:
            clipped.append(current)

        previous = current
        previous_inside = current_inside

    return clipped


def clip_polygon_to_tile(
    polygon: np.ndarray,
    window: TileWindow,
) -> np.ndarray | None:
    """
    Clips a polygon to the visible portion of a tile.

    The clipped polygon is converted back to a 4-point oriented rectangle so it
    can be written in YOLO OBB format.
    """
    local_points = [point.astype(np.float32) - np.array([window.x, window.y], dtype=np.float32) for point in polygon]
    local_points = _clip_against_vertical(local_points, 0.0, keep_greater=True)
    local_points = _clip_against_vertical(local_points, float(window.width), keep_greater=False)
    local_points = _clip_against_horizontal(local_points, 0.0, keep_greater=True)
    local_points = _clip_against_horizontal(local_points, float(window.height), keep_greater=False)

    if len(local_points) < 3:
        return None

    contour = np.array(local_points, dtype=np.float32)
    if cv2.contourArea(contour) <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box[:, 0] = np.clip(box[:, 0], 0, window.width)
    box[:, 1] = np.clip(box[:, 1], 0, window.height)
    return box.astype(np.float32)


def normalize_polygon(polygon: np.ndarray, tile_size: int) -> list[float]:
    """Normalizes polygon coordinates to the YOLO OBB range."""
    normalized: list[float] = []
    clipped = np.clip(polygon, 0, tile_size)
    for x_coord, y_coord in clipped:
        normalized.append(round(float(x_coord) / tile_size, 6))
        normalized.append(round(float(y_coord) / tile_size, 6))
    return normalized


def build_tile_annotations(
    objects: list[SourceObject],
    window: TileWindow,
    tile_size: int,
) -> list[tuple[int, list[float]]]:
    """Builds normalized YOLO OBB annotations for a single tile."""
    annotations: list[tuple[int, list[float]]] = []

    for source_object in objects:
        polygon = np.array(source_object.polygon, dtype=np.float32)
        clipped_polygon = clip_polygon_to_tile(polygon, window)
        if clipped_polygon is None:
            continue

        annotations.append((source_object.class_id, normalize_polygon(clipped_polygon, tile_size)))

    return annotations


def deterministic_keep_empty_tile(
    image_stem: str,
    tile_index: int,
    keep_ratio: float,
    random_seed: int,
) -> bool:
    """Keeps empty tiles reproducibly without relying on global RNG state."""
    if keep_ratio <= 0:
        return False
    if keep_ratio >= 1:
        return True

    digest = hashlib.sha256(
        f"{random_seed}:{image_stem}:{tile_index}".encode()
    ).hexdigest()
    threshold = int(digest[:8], 16) / 0xFFFFFFFF
    return threshold <= keep_ratio


def build_tiled_split(
    source_images_dir: Path,
    source_labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    class_names: list[str],
    tile_size: int,
    overlap: int,
    empty_keep_ratio: float,
    random_seed: int,
) -> dict[str, int]:
    """Builds a tiled dataset split from DOTA-style source assets."""
    class_to_id = {name: index for index, name in enumerate(class_names)}
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images": 0,
        "source_objects": 0,
        "tiles_generated": 0,
        "tiles_saved": 0,
        "empty_tiles_saved": 0,
        "objects_written": 0,
    }

    for image_path in list_image_paths(source_images_dir):
        label_path = source_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing source label file for {image_path.name}: {label_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read source image: {image_path}")

        objects = read_dota_label(label_path, class_to_id=class_to_id)
        windows = generate_tile_windows(image.shape[1], image.shape[0], tile_size, overlap)
        stats["source_images"] += 1
        stats["source_objects"] += len(objects)
        stats["tiles_generated"] += len(windows)

        for window in windows:
            tile_annotations = build_tile_annotations(objects, window, tile_size)
            if not tile_annotations and not deterministic_keep_empty_tile(
                image_stem=image_path.stem,
                tile_index=window.index,
                keep_ratio=empty_keep_ratio,
                random_seed=random_seed,
            ):
                continue

            tile_name = f"{image_path.stem}_{window.index:04d}"
            tile_image_path = output_images_dir / f"{tile_name}.png"
            tile_label_path = output_labels_dir / f"{tile_name}.txt"
            tile_image = extract_padded_tile(image, window, tile_size)

            if not cv2.imwrite(str(tile_image_path), tile_image):
                raise RuntimeError(f"Unable to write tiled image: {tile_image_path}")

            label_lines = [
                str(class_id) + " " + " ".join(f"{coord:.6f}" for coord in coords)
                for class_id, coords in tile_annotations
            ]
            tile_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            stats["tiles_saved"] += 1
            stats["objects_written"] += len(tile_annotations)
            if not tile_annotations:
                stats["empty_tiles_saved"] += 1

    return stats
