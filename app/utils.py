"""
Utilitaires pour le traitement d'images et les opérations géométriques.
"""

from collections.abc import Generator
from typing import Optional

import cv2
import numpy as np


def _generate_axis_starts(image_size: int, tile_size: int, overlap: int) -> list[int]:
    """Calcule des positions de départ sans doublons pour couvrir complètement un axe."""
    if image_size <= tile_size:
        return [0]

    step = tile_size - overlap
    max_start = image_size - tile_size
    num_tiles = int(np.ceil(max_start / step)) + 1

    starts = [int(round(index * max_start / (num_tiles - 1))) for index in range(num_tiles)]

    deduplicated_starts = []
    for start in starts:
        if not deduplicated_starts or start != deduplicated_starts[-1]:
            deduplicated_starts.append(start)

    return deduplicated_starts


def _compute_valid_axis_bounds(
    starts: list[int],
    index: int,
    image_size: int,
    tile_size: int,
) -> tuple[float, float]:
    """Découpe l'overlap réel au milieu entre deux tiles voisins."""
    start = starts[index]
    end = min(start + tile_size, image_size)

    if index == 0:
        valid_min = float(start)
    else:
        prev_end = min(starts[index - 1] + tile_size, image_size)
        valid_min = (prev_end + start) / 2.0

    if index == len(starts) - 1:
        valid_max = float(end)
    else:
        next_start = starts[index + 1]
        valid_max = (end + next_start) / 2.0

    return valid_min, valid_max


def generate_tiles(
    image_width: int, image_height: int, tile_size: int, overlap: int
) -> Generator[tuple[int, int, int, int], None, None]:
    """
    Génère les coordonnées des tiles avec overlap.

    Yields:
        (x_start, y_start, x_end, y_end) pour chaque tile
    """
    x_starts = _generate_axis_starts(image_width, tile_size, overlap)
    y_starts = _generate_axis_starts(image_height, tile_size, overlap)

    for y_start in y_starts:
        y_end = min(y_start + tile_size, image_height)
        for x_start in x_starts:
            x_end = min(x_start + tile_size, image_width)
            yield (x_start, y_start, x_end, y_end)


def generate_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> Generator[tuple[tuple[int, int, int, int], tuple[float, float, float, float]], None, None]:
    """
    Génère chaque tile avec sa zone valide réelle.

    La zone valide partage les overlaps au milieu entre les tiles adjacents.
    """
    x_starts = _generate_axis_starts(image_width, tile_size, overlap)
    y_starts = _generate_axis_starts(image_height, tile_size, overlap)

    for y_index, y_start in enumerate(y_starts):
        y_end = min(y_start + tile_size, image_height)
        valid_y_min, valid_y_max = _compute_valid_axis_bounds(
            y_starts,
            y_index,
            image_height,
            tile_size,
        )

        for x_index, x_start in enumerate(x_starts):
            x_end = min(x_start + tile_size, image_width)
            valid_x_min, valid_x_max = _compute_valid_axis_bounds(
                x_starts,
                x_index,
                image_width,
                tile_size,
            )

            yield (
                (x_start, y_start, x_end, y_end),
                (valid_x_min, valid_y_min, valid_x_max, valid_y_max),
            )


def extract_tile(image: np.ndarray, coords: tuple[int, int, int, int]) -> np.ndarray:
    """Extrait un tile d'une image."""
    x_start, y_start, x_end, y_end = coords
    return image[y_start:y_end, x_start:x_end].copy()


def reproject_obb_to_global(
    obb_points: np.ndarray, tile_offset_x: int, tile_offset_y: int
) -> np.ndarray:
    """
    Reprojette les points OBB des coordonnées tile vers les coordonnées globales.

    Args:
        obb_points: Array de shape (4, 2) avec les 4 coins du polygone
        tile_offset_x: Offset X du tile dans l'image globale
        tile_offset_y: Offset Y du tile dans l'image globale

    Returns:
        Points reprojetés dans le repère global
    """
    global_points = obb_points.copy()
    global_points[:, 0] += tile_offset_x
    global_points[:, 1] += tile_offset_y
    return global_points


def obb_to_aabb(obb_points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convertit un OBB en Axis-Aligned Bounding Box pour le NMS.

    Args:
        obb_points: Array de shape (4, 2)

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x_min = np.min(obb_points[:, 0])
    y_min = np.min(obb_points[:, 1])
    x_max = np.max(obb_points[:, 0])
    y_max = np.max(obb_points[:, 1])
    return (x_min, y_min, x_max, y_max)


def compute_iou_aabb(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    """
    Calcule l'IoU entre deux AABB.

    Args:
        box1, box2: (x_min, y_min, x_max, y_max)

    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def nms_obb(detections: list[dict], iou_threshold: float) -> list[dict]:
    """
    Applique un NMS global sur les détections OBB.
    Utilise une approximation AABB pour le calcul d'IoU.

    Args:
        detections: Liste de dicts avec 'polygon', 'confidence', 'class_id'
        iou_threshold: Seuil IoU pour la suppression

    Returns:
        Liste filtrée de détections
    """
    if not detections:
        return []

    # Trier par confiance décroissante
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    # Pré-calculer les AABB
    aabbs = [obb_to_aabb(np.array(d["polygon"])) for d in sorted_dets]

    keep = []
    suppressed = set()

    for i, det in enumerate(sorted_dets):
        if i in suppressed:
            continue

        keep.append(det)

        # Supprimer les détections avec IoU élevé ET même classe
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue

            # NMS intra-classe uniquement
            if sorted_dets[j]["class_id"] != det["class_id"]:
                continue

            iou = compute_iou_aabb(aabbs[i], aabbs[j])
            if iou > iou_threshold:
                suppressed.add(j)

    return keep


def is_detection_in_padding(
    obb_points: np.ndarray,
    tile_coords: Optional[tuple[int, int, int, int]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    overlap: Optional[int] = None,
    valid_region: Optional[tuple[float, float, float, float]] = None,
) -> bool:
    """
    Vérifie si une détection est entièrement dans la zone de padding/overlap.

    On garde une détection si son centre est dans la zone "principale" du tile,
    c'est-à-dire pas dans les marges d'overlap (sauf si on est au bord de l'image).

    Args:
        obb_points: Points du polygone
        tile_coords: (x_start, y_start, x_end, y_end) du tile
        image_width: Largeur totale de l'image
        image_height: Hauteur totale de l'image
        overlap: Taille de l'overlap
        valid_region: Zone valide réelle (x_min, y_min, x_max, y_max)

    Returns:
        True si la détection doit être ignorée (dans le padding)
    """
    # Calculer le centre du polygone
    center_x = np.mean(obb_points[:, 0])
    center_y = np.mean(obb_points[:, 1])

    if valid_region is None:
        if tile_coords is None or image_width is None or image_height is None or overlap is None:
            raise ValueError(
                "tile_coords/image_width/image_height/overlap ou valid_region sont requis"
            )

        x_start, y_start, x_end, y_end = tile_coords
        half_overlap = overlap // 2
        valid_x_min = x_start + (half_overlap if x_start > 0 else 0)
        valid_y_min = y_start + (half_overlap if y_start > 0 else 0)
        valid_x_max = x_end - (half_overlap if x_end < image_width else 0)
        valid_y_max = y_end - (half_overlap if y_end < image_height else 0)
    else:
        valid_x_min, valid_y_min, valid_x_max, valid_y_max = valid_region

    # Le centre doit être dans la zone valide
    in_valid_zone = valid_x_min <= center_x < valid_x_max and valid_y_min <= center_y < valid_y_max

    return not in_valid_zone


def draw_obb_on_image(
    image: np.ndarray, detections: list[dict], class_colors: dict, thickness: int = 2
) -> np.ndarray:
    """
    Dessine les OBB sur une image.

    Args:
        image: Image BGR
        detections: Liste de détections avec 'polygon', 'class_name', 'confidence'
        class_colors: Dict mapping class_id -> couleur BGR
        thickness: Épaisseur des lignes

    Returns:
        Image annotée
    """
    annotated = image.copy()

    for det in detections:
        points = np.array(det["polygon"], dtype=np.int32)
        class_id = det["class_id"]
        color = class_colors.get(class_id, (255, 255, 255))

        # Dessiner le polygone
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=thickness)

        # Ajouter le label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        label_pos = (int(points[0, 0]), int(points[0, 1]) - 5)

        # Fond pour le texte
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (label_pos[0], label_pos[1] - text_h - 4),
            (label_pos[0] + text_w, label_pos[1] + 2),
            color,
            -1,
        )

        # Texte
        cv2.putText(
            annotated,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
