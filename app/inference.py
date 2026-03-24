"""
Module d'inférence pour la détection OBB avec stratégie de tiling.
Thread-safe avec gestion de la mémoire GPU.
"""

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ultralytics import YOLO

from app.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    NMS_IOU_THRESHOLD,
    OUTPUTS_DIR,
    OVERLAP,
    TILE_SIZE,
    logger,
)
from app.schemas import Detection, PredictionResponse
from app.utils import (
    draw_obb_on_image,
    extract_tile,
    generate_tile_grid,
    is_detection_in_padding,
    nms_obb,
    reproject_obb_to_global,
)


class OBBDetector:
    """
    Détecteur OBB avec stratégie de tiling pour images aériennes.
    Thread-safe grâce à un lock interne.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialise le détecteur.

        Args:
            model_path: Chemin vers le modèle YOLO. Utilise MODEL_PATH par défaut.
        """
        self.model_path = model_path or MODEL_PATH
        self.model: Optional[YOLO] = None
        self.tile_size = TILE_SIZE
        self.overlap = OVERLAP
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.nms_threshold = NMS_IOU_THRESHOLD
        self.class_names = CLASS_NAMES

        # Lock pour thread-safety (YOLO n'est pas thread-safe)
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """Charge le modèle YOLO."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

        logger.info(f"Chargement du modèle depuis {self.model_path}")
        self.model = YOLO(str(self.model_path))
        logger.info("Modèle chargé avec succès")

    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self.model is not None

    def _clear_gpu_memory(self) -> None:
        """Libère la mémoire GPU si disponible."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_tile(self, tile: np.ndarray) -> list[dict]:
        """
        Lance l'inférence sur un tile unique.
        ATTENTION: Cette méthode doit être appelée avec le lock acquis.

        Args:
            tile: Image du tile (BGR)

        Returns:
            Liste de détections brutes avec coordonnées locales au tile
        """
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")

        results = self.model.predict(tile, conf=self.conf_threshold, verbose=False, task="obb")

        detections = []

        for result in results:
            if result.obb is None:
                continue

            boxes = result.obb
            for i in range(len(boxes)):
                # Extraire les 4 points du polygone OBB
                if hasattr(boxes, "xyxyxyxy") and boxes.xyxyxyxy is not None:
                    points = boxes.xyxyxyxy[i].cpu().numpy()
                else:
                    logger.warning(f"Format OBB non supporté, attributs disponibles: {dir(boxes)}")
                    continue

                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append(
                    {
                        "polygon": points.reshape(4, 2),
                        "confidence": conf,
                        "class_id": cls_id,
                    }
                )

        return detections

    def predict_full_image(self, image: np.ndarray) -> tuple[list[dict], int, int]:
        """
        Pipeline complet de prédiction sur une image.
        Thread-safe grâce au lock interne.

        1. Découpe l'image en tiles avec overlap
        2. Lance la prédiction sur chaque tile
        3. Reprojette les coordonnées vers le repère global
        4. Filtre les détections dans les zones de padding
        5. Applique le NMS global

        Args:
            image: Image complète BGR

        Returns:
            Tuple (liste_detections, largeur_image, hauteur_image)
        """
        if not self.is_loaded():
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")

        height, width = image.shape[:2]
        all_detections = []

        # Acquérir le lock pour toute la durée de l'inférence
        with self._lock:
            try:
                # Générer et traiter chaque tile
                for tile_coords, valid_region in generate_tile_grid(
                    width,
                    height,
                    self.tile_size,
                    self.overlap,
                ):
                    x_start, y_start, x_end, y_end = tile_coords

                    # Extraire le tile
                    tile = extract_tile(image, tile_coords)

                    # Prédiction sur le tile
                    tile_detections = self.predict_tile(tile)

                    # Reprojeter et filtrer
                    for det in tile_detections:
                        # Reprojeter vers coordonnées globales
                        global_polygon = reproject_obb_to_global(det["polygon"], x_start, y_start)

                        # Vérifier si la détection est dans la zone de padding
                        if is_detection_in_padding(
                            global_polygon,
                            valid_region=valid_region,
                        ):
                            continue

                        all_detections.append(
                            {
                                "polygon": global_polygon.tolist(),
                                "confidence": det["confidence"],
                                "class_id": det["class_id"],
                            }
                        )

            finally:
                # Libérer la mémoire GPU après traitement
                self._clear_gpu_memory()

        # Appliquer le NMS global (hors du lock car pas besoin du modèle)
        filtered_detections = nms_obb(all_detections, self.nms_threshold)

        logger.debug(
            f"Image {width}x{height}: {len(all_detections)} détections brutes, {len(filtered_detections)} après NMS"
        )

        return filtered_detections, width, height

    def format_detections(self, detections: list[dict]) -> list[Detection]:
        """
        Formate les détections brutes en objets Detection.

        Args:
            detections: Liste de dicts avec polygon, confidence, class_id

        Returns:
            Liste d'objets Detection
        """
        formatted = []
        for det in detections:
            formatted.append(
                Detection(
                    class_id=det["class_id"],
                    class_name=self.class_names.get(det["class_id"], "unknown"),
                    confidence=round(det["confidence"], 4),
                    polygon=det["polygon"],
                )
            )
        return formatted

    def predict(self, image: np.ndarray) -> PredictionResponse:
        """
        Interface principale de prédiction.

        Args:
            image: Image BGR

        Returns:
            PredictionResponse avec toutes les détections
        """
        detections, width, height = self.predict_full_image(image)
        formatted = self.format_detections(detections)

        return PredictionResponse(
            image_width=width,
            image_height=height,
            nb_detections=len(formatted),
            detections=formatted,
        )

    def predict_and_save(
        self, image: np.ndarray, output_filename: str
    ) -> tuple[PredictionResponse, str]:
        """
        Prédit et sauvegarde l'image annotée.

        Args:
            image: Image BGR
            output_filename: Nom du fichier de sortie

        Returns:
            Tuple (PredictionResponse, chemin_fichier_sauvegardé)
        """
        detections, width, height = self.predict_full_image(image)
        formatted = self.format_detections(detections)

        # Annoter l'image
        annotated = draw_obb_on_image(
            image,
            [
                {
                    "polygon": d.polygon,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                }
                for d in formatted
            ],
            CLASS_COLORS,
        )

        # Sauvegarder
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        outputs_dir = OUTPUTS_DIR.resolve()
        output_path = (OUTPUTS_DIR / output_filename).resolve()

        if output_path.parent != outputs_dir:
            raise ValueError("Le fichier de sortie doit rester dans le dossier de predictions")

        write_success = cv2.imwrite(str(output_path), annotated)
        if not write_success:
            raise RuntimeError(f"Echec de la sauvegarde de l'image annotee: {output_path}")

        logger.info(f"Image annotée sauvegardée: {output_path}")

        response = PredictionResponse(
            image_width=width,
            image_height=height,
            nb_detections=len(formatted),
            detections=formatted,
        )

        return response, str(output_path)


# Instance globale du détecteur (chargée au démarrage)
detector = OBBDetector()
