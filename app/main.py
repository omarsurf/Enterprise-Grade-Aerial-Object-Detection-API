"""
FastAPI API for aerial OBB detection with basic runtime hardening controls.
"""

from __future__ import annotations

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
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.artifacts import build_model_info
from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    BUSY_RETRY_AFTER_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_INFLIGHT_PREDICTIONS,
    MODEL_PATH,
    RATE_LIMIT_MODEL_INFO,
    RATE_LIMIT_PREDICT,
    RATE_LIMIT_PREDICT_AND_SAVE,
    HealthStatus,
    logger,
)
from app.inference import detector
from app.observability import install_observability
from app.rate_limit import Limiter, RateLimitExceeded
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionWithSaveResponse,
)
from app.security import AuthContext, get_rate_limit_key, require_api_key

_executor = ThreadPoolExecutor(max_workers=MAX_INFLIGHT_PREDICTIONS)
_prediction_semaphore = asyncio.Semaphore(MAX_INFLIGHT_PREDICTIONS)
limiter = Limiter(key_func=get_rate_limit_key, default_limits=[], headers_enabled=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Loads the model on startup and stores the promoted model version for logs.
    """
    logger.info("Starting API - loading model from %s", MODEL_PATH)
    try:
        detector.load_model()
        logger.info("Model loaded successfully - API ready")
    except FileNotFoundError as exc:
        detector.model = None
        logger.error("CRITICAL MODEL ERROR: %s", exc)
        logger.warning("API will start in degraded mode - predictions will fail")
    except Exception as exc:  # pragma: no cover - exercised in async lifespan test
        detector.model = None
        logger.exception("CRITICAL MODEL ERROR during load: %s", exc)
        logger.warning("API will start in degraded mode - predictions will fail")

    runtime_info = build_model_info(model_loaded=detector.is_loaded())
    app.state.model_version = runtime_info.get("model_version")

    yield

    logger.info("Stopping API - cleaning resources")
    _executor.shutdown(wait=True)


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)
app.state.limiter = limiter
install_observability(app)


def validate_file(file: UploadFile) -> None:
    """
    Validates the uploaded file extension and MIME type.

    Some clients send incorrect MIME types, so extension validation is authoritative.
    """
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Extension '{ext}' non supportee. "
                f"Extensions acceptees: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning("Unexpected MIME type %s for %s", file.content_type, filename)


async def read_file_with_limit(file: UploadFile) -> bytes:
    """Reads the uploaded file while enforcing the configured max size."""
    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Fichier trop volumineux. Taille max: {MAX_FILE_SIZE_MB}MB",
        )

    return contents


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decodes an uploaded image from raw bytes."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de decoder l'image. Verifiez que le fichier est valide.",
        )

    return image


def check_model_loaded() -> None:
    """Ensures the detector is loaded before accepting prediction work."""
    if not detector.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporairement indisponible. Le modele n'est pas charge.",
        )


def build_output_filename(original_name: str) -> str:
    """Builds a safe output filename derived from the uploaded image name."""
    safe_name = (original_name or "image").replace("\\", "/").rsplit("/", 1)[-1]
    base_name = Path(safe_name).stem
    sanitized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")

    if not sanitized_name:
        sanitized_name = "image"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{sanitized_name}_{timestamp}_{unique_id}.png"


def get_runtime_model_info() -> ModelInfoResponse:
    """Builds the runtime `model-info` payload from the promoted artifact bundle."""
    payload = ModelInfoResponse(**build_model_info(model_loaded=detector.is_loaded()))
    app.state.model_version = payload.model_version
    return payload


@asynccontextmanager
async def prediction_slot():
    """Provides a fail-fast concurrency slot for expensive inference requests."""
    try:
        await asyncio.wait_for(_prediction_semaphore.acquire(), timeout=0.001)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference capacity exhausted. Retry later.",
            headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
        ) from exc

    try:
        yield
    finally:
        _prediction_semaphore.release()


def _error_payload(status_code: int, detail: str) -> dict[str, object]:
    return {"error": detail, "status_code": status_code}


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Model unavailable"}},
    summary="Health Check",
    description="Checks the health of the API and the model slot.",
)
async def health_check(response: Response) -> HealthResponse:
    """Returns the service health without requiring authentication."""
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
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Model Info",
    description="Exposes the metadata and traceability of the promoted model.",
)
@limiter.limit(RATE_LIMIT_MODEL_INFO)
async def model_info(
    request: Request,
    _auth: Annotated[AuthContext, Depends(require_api_key)],
) -> ModelInfoResponse:
    """Returns runtime metadata for the currently served model."""
    return get_runtime_model_info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Service unavailable or saturated"},
    },
    summary="OBB Prediction",
    description="Runs tiled OBB detection on an uploaded aerial image.",
)
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Image to analyze (PNG, JPG, TIFF)")],
    _auth: Annotated[AuthContext, Depends(require_api_key)],
) -> PredictionResponse:
    """Runs prediction and returns detections as JSON."""
    check_model_loaded()
    validate_file(file)

    contents = await read_file_with_limit(file)
    image = decode_image(contents)

    async with prediction_slot():
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, detector.predict, image)

    return result


@app.post(
    "/predict-and-save",
    response_model=PredictionWithSaveResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Service unavailable or saturated"},
    },
    summary="OBB Prediction With Save",
    description="Runs tiled OBB detection and saves an annotated image.",
)
@limiter.limit(RATE_LIMIT_PREDICT_AND_SAVE)
async def predict_and_save(
    request: Request,
    file: Annotated[UploadFile, File(description="Image to analyze (PNG, JPG, TIFF)")],
    _auth: Annotated[AuthContext, Depends(require_api_key)],
) -> PredictionWithSaveResponse:
    """Runs prediction and persists an annotated image under outputs/predictions."""
    check_model_loaded()
    validate_file(file)

    contents = await read_file_with_limit(file)
    image = decode_image(contents)
    output_filename = build_output_filename(file.filename or "image")

    async with prediction_slot():
        loop = asyncio.get_running_loop()
        result, output_path = await loop.run_in_executor(
            _executor,
            detector.predict_and_save,
            image,
            output_filename,
        )

    return PredictionWithSaveResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        nb_detections=result.nb_detections,
        detections=result.detections,
        output_path=output_path,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Preserves HTTP status codes while returning the standard JSON envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.status_code, str(exc.detail)),
        headers=exc.headers,
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Returns JSON errors when the in-memory rate limit is exceeded."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=_error_payload(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded."),
        headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions without masking explicit HTTP errors."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(status.HTTP_500_INTERNAL_SERVER_ERROR, "Erreur interne du serveur"),
    )
