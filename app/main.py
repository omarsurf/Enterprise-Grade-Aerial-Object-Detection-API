"""
API FastAPI pour la détection OBB sur images aériennes.
"""

import asyncio
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse

from app.artifacts import build_model_info
from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MODEL_PATH,
    HealthStatus,
    logger,
)
from app.inference import detector
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionWithSaveResponse,
)

# ThreadPoolExecutor pour exécuter l'inférence sans bloquer l'event loop.
# Le lock du détecteur sérialise l'accès au modèle.
_executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.
    Charge le modèle au démarrage.
    """
    logger.info(f"Démarrage de l'API - Chargement du modèle depuis {MODEL_PATH}")
    try:
        detector.load_model()
        logger.info("Modèle chargé avec succès - API prête")
    except FileNotFoundError as e:
        detector.model = None
        logger.error(f"ERREUR CRITIQUE: {e}")
        logger.warning("L'API démarrera en mode dégradé - les prédictions échoueront")
    except Exception as e:  # pragma: no cover - exercised via async lifespan test
        detector.model = None
        logger.exception(f"ERREUR CRITIQUE au chargement du modèle: {e}")
        logger.warning("L'API démarrera en mode dégradé - les prédictions échoueront")

    yield

    logger.info("Arrêt de l'API - Nettoyage des ressources")
    _executor.shutdown(wait=True)


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


# === Utilitaires ===


def validate_file(file: UploadFile) -> None:
    """
    Valide le fichier uploadé (extension et MIME type).

    Args:
        file: Fichier uploadé

    Raises:
        HTTPException: Si le fichier n'est pas valide
    """
    # Vérifier l'extension
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extension '{ext}' non supportée. Extensions acceptées: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Vérifier le MIME type (si disponible)
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"MIME type inattendu: {file.content_type} pour {filename}")
        # On laisse passer avec un warning car certains clients n'envoient pas le bon MIME type


async def read_file_with_limit(file: UploadFile) -> bytes:
    """
    Lit le fichier avec une limite de taille.

    Args:
        file: Fichier uploadé

    Returns:
        Contenu du fichier en bytes

    Raises:
        HTTPException: Si le fichier est trop grand
    """
    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Fichier trop volumineux. Taille max: {MAX_FILE_SIZE_MB}MB",
        )

    return contents


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Décode une image depuis des bytes.

    Args:
        file_bytes: Contenu du fichier image

    Returns:
        Image BGR numpy array

    Raises:
        HTTPException: Si l'image ne peut pas être décodée
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de décoder l'image. Vérifiez que le fichier est une image valide.",
        )

    return image


def check_model_loaded() -> None:
    """
    Vérifie que le modèle est chargé.

    Raises:
        HTTPException: Si le modèle n'est pas chargé
    """
    if not detector.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporairement indisponible. Le modèle n'est pas chargé.",
        )


def build_output_filename(original_name: str) -> str:
    """Construit un nom de fichier de sortie sûr à partir du nom uploadé."""
    safe_name = (original_name or "image").replace("\\", "/").rsplit("/", 1)[-1]
    base_name = Path(safe_name).stem
    sanitized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")

    if not sanitized_name:
        sanitized_name = "image"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{sanitized_name}_{timestamp}_{unique_id}.png"


def get_runtime_model_info() -> ModelInfoResponse:
    """Construit la réponse `model-info` à partir du bundle promu."""
    return ModelInfoResponse(**build_model_info(model_loaded=detector.is_loaded()))


# === Endpoints ===


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Modèle indisponible"}},
    summary="Health Check",
    description="Vérifie l'état de l'API et du modèle.",
)
async def health_check(response: Response) -> HealthResponse:
    """Retourne l'état de santé de l'API."""
    is_loaded = detector.is_loaded()
    response.status_code = status.HTTP_200_OK if is_loaded else status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=HealthStatus.HEALTHY if is_loaded else HealthStatus.DEGRADED,
        model_loaded=is_loaded,
        model_path=str(MODEL_PATH),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Info",
    description="Expose les métadonnées du modèle promu servi par l'API.",
)
async def model_info() -> ModelInfoResponse:
    """Retourne l'identité et la traçabilité du modèle servi."""
    return get_runtime_model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Image invalide"},
        413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
        503: {"model": ErrorResponse, "description": "Service indisponible"},
    },
    summary="Prédiction OBB",
    description="Détecte les objets orientés dans une image aérienne.",
)
async def predict(
    file: Annotated[UploadFile, File(description="Image à analyser (PNG, JPG, TIFF)")],
) -> PredictionResponse:
    """
    Endpoint de prédiction standard.

    - Accepte une image uploadée (max 50MB)
    - Retourne les détections au format JSON
    """
    check_model_loaded()
    validate_file(file)

    contents = await read_file_with_limit(file)
    image = decode_image(contents)

    # Exécuter l'inférence dans un thread séparé pour ne pas bloquer l'event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, detector.predict, image)

    return result


@app.post(
    "/predict-and-save",
    response_model=PredictionWithSaveResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Image invalide"},
        413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
        503: {"model": ErrorResponse, "description": "Service indisponible"},
    },
    summary="Prédiction OBB avec sauvegarde",
    description="Détecte les objets et sauvegarde une image annotée.",
)
async def predict_and_save(
    file: Annotated[UploadFile, File(description="Image à analyser (PNG, JPG, TIFF)")],
) -> PredictionWithSaveResponse:
    """
    Endpoint de prédiction avec sauvegarde de l'image annotée.

    - Accepte une image uploadée (max 50MB)
    - Sauvegarde l'image avec les détections dessinées
    - Retourne les détections + le chemin du fichier sauvegardé
    """
    check_model_loaded()
    validate_file(file)

    contents = await read_file_with_limit(file)
    image = decode_image(contents)

    # Générer un nom de fichier unique et sûr
    output_filename = build_output_filename(file.filename or "image")

    # Exécuter l'inférence dans un thread séparé
    loop = asyncio.get_running_loop()
    result, output_path = await loop.run_in_executor(
        _executor, detector.predict_and_save, image, output_filename
    )

    return PredictionWithSaveResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        nb_detections=result.nb_detections,
        detections=result.detections,
        output_path=output_path,
    )


# === Exception Handlers ===


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Gestionnaire pour les HTTPException - préserve le status code."""
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Gestionnaire d'erreurs global pour les exceptions non gérées.
    Ne masque PAS les HTTPException (gérées séparément).
    """
    logger.exception(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Erreur interne du serveur", "status_code": 500},
    )
