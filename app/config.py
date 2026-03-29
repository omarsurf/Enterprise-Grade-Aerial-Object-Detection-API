"""
Configuration centralisée pour l'API de détection OBB.
Supporte les variables d'environnement pour la personnalisation.
"""

import logging
import os
from enum import Enum
from pathlib import Path

# === Logging ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("aerial-obb-api")

# === Chemins ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "predictions"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROMOTED_ARTIFACTS_DIR = ARTIFACTS_DIR / "promoted"
DATASET_ARTIFACTS_DIR = ARTIFACTS_DIR / "datasets"
PROMOTED_MANIFEST_PATH = PROMOTED_ARTIFACTS_DIR / "manifest.json"
PROMOTED_METRICS_PATH = PROMOTED_ARTIFACTS_DIR / "metrics.json"
PROMOTED_PARAMS_PATH = PROMOTED_ARTIFACTS_DIR / "params.json"

# Créer les dossiers utiles s'ils n'existent pas
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROMOTED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# === Modèle ===
def resolve_model_path() -> Path:
    """Résout le chemin du modèle à partir de l'env ou du poids tiled attendu."""
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return Path(model_path)

    return MODELS_DIR / "best_tiled.pt"


MODEL_PATH = resolve_model_path()

# === Paramètres d'inférence (configurables via env vars) ===
TILE_SIZE: int = int(os.getenv("TILE_SIZE", "1024"))
OVERLAP: int = int(os.getenv("OVERLAP", "200"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
NMS_IOU_THRESHOLD: float = float(os.getenv("NMS_IOU_THRESHOLD", "0.5"))
MAX_INFLIGHT_PREDICTIONS: int = int(os.getenv("MAX_INFLIGHT_PREDICTIONS", "2"))
BUSY_RETRY_AFTER_SECONDS: int = int(os.getenv("BUSY_RETRY_AFTER_SECONDS", "15"))

# === Sécurité / observabilité ===
API_KEY_HEADER_NAME = "X-API-Key"
REQUEST_ID_HEADER_NAME = "X-Request-ID"
DEFAULT_DEV_API_KEY = "local-dev-key"
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "10/minute")
RATE_LIMIT_PREDICT_AND_SAVE = os.getenv("RATE_LIMIT_PREDICT_AND_SAVE", "5/minute")
RATE_LIMIT_MODEL_INFO = os.getenv("RATE_LIMIT_MODEL_INFO", "30/minute")

# === Limites de fichier ===
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
ALLOWED_MIME_TYPES: set = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/x-tiff",
}

# === Classes ===
CLASS_NAMES: dict[int, str] = {
    0: "plane",
    1: "ship",
    2: "small-vehicle",
    3: "large-vehicle",
}

# === Couleurs pour l'annotation (BGR) ===
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),  # Bleu pour plane
    1: (0, 255, 0),  # Vert pour ship
    2: (0, 255, 255),  # Jaune pour small-vehicle
    3: (0, 0, 255),  # Rouge pour large-vehicle
}


# === Health Status ===
class HealthStatus(str, Enum):
    """Status de santé de l'API."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# === API ===
API_TITLE = "Aerial OBB Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API de détection d'objets orientés (OBB) sur images aériennes.

Utilise un modèle YOLO OBB entraîné avec une stratégie de tiling
pour détecter : avions, navires, petits et grands véhicules.
"""


# === Validation au démarrage ===
def validate_config() -> None:
    """Valide la configuration au démarrage."""
    if OVERLAP >= TILE_SIZE:
        raise ValueError(f"OVERLAP ({OVERLAP}) doit être < TILE_SIZE ({TILE_SIZE})")

    if CONFIDENCE_THRESHOLD < 0 or CONFIDENCE_THRESHOLD > 1:
        raise ValueError(f"CONFIDENCE_THRESHOLD doit être entre 0 et 1, got {CONFIDENCE_THRESHOLD}")

    if NMS_IOU_THRESHOLD < 0 or NMS_IOU_THRESHOLD > 1:
        raise ValueError(f"NMS_IOU_THRESHOLD doit être entre 0 et 1, got {NMS_IOU_THRESHOLD}")

    if MAX_INFLIGHT_PREDICTIONS < 1:
        raise ValueError(
            f"MAX_INFLIGHT_PREDICTIONS doit être >= 1, got {MAX_INFLIGHT_PREDICTIONS}"
        )

    if BUSY_RETRY_AFTER_SECONDS < 1:
        raise ValueError(
            f"BUSY_RETRY_AFTER_SECONDS doit être >= 1, got {BUSY_RETRY_AFTER_SECONDS}"
        )

    # Vérifier que toutes les classes ont une couleur
    for class_id in CLASS_NAMES:
        if class_id not in CLASS_COLORS:
            raise ValueError(
                f"Classe {class_id} ({CLASS_NAMES[class_id]}) n'a pas de couleur définie"
            )

    logger.info(
        "Configuration validée: TILE_SIZE=%s, OVERLAP=%s, CONF=%s, MAX_INFLIGHT=%s",
        TILE_SIZE,
        OVERLAP,
        CONFIDENCE_THRESHOLD,
        MAX_INFLIGHT_PREDICTIONS,
    )


# Valider au chargement du module
validate_config()
