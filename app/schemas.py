"""
Schémas Pydantic pour la validation des données d'entrée/sortie.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import HealthStatus


class Detection(BaseModel):
    """Une détection OBB unique."""

    class_id: int = Field(..., ge=0, description="ID de la classe détectée")
    class_name: str = Field(..., min_length=1, description="Nom de la classe détectée")
    confidence: float = Field(..., ge=0, le=1, description="Score de confiance")
    polygon: list[list[float]] = Field(
        ..., description="4 points du polygone OBB [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, v: list[list[float]]) -> list[list[float]]:
        """Valide que le polygone a exactement 4 points avec coordonnées valides."""
        if len(v) != 4:
            raise ValueError(f"Le polygone doit avoir exactement 4 points, reçu {len(v)}")

        for i, point in enumerate(v):
            if len(point) != 2:
                raise ValueError(f"Le point {i} doit avoir 2 coordonnées (x, y), reçu {len(point)}")
            if not all(isinstance(coord, (int, float)) for coord in point):
                raise ValueError(f"Les coordonnées du point {i} doivent être numériques")
            # Note: Les coordonnées négatives sont possibles avec le tiling
            # car les détections peuvent déborder légèrement hors de l'image

        return v


class PredictionResponse(BaseModel):
    """Réponse standard pour une prédiction."""

    image_width: int = Field(..., gt=0, description="Largeur de l'image en pixels")
    image_height: int = Field(..., gt=0, description="Hauteur de l'image en pixels")
    nb_detections: int = Field(..., ge=0, description="Nombre total de détections")
    detections: list[Detection] = Field(default_factory=list)


class PredictionWithSaveResponse(PredictionResponse):
    """Réponse pour une prédiction avec sauvegarde d'image annotée."""

    output_path: str = Field(..., min_length=1, description="Chemin du fichier image sauvegardé")


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: HealthStatus = Field(default=HealthStatus.HEALTHY, description="État de santé de l'API")
    model_loaded: bool = Field(..., description="Indique si le modèle est chargé")
    model_path: str = Field(..., description="Chemin du modèle utilisé")


class ModelInfoResponse(BaseModel):
    """Réponse décrivant le modèle promu servi par l'API."""

    model_loaded: bool = Field(..., description="Indique si le modèle est chargé en mémoire")
    manifest_available: bool = Field(
        ..., description="Indique si un manifeste promu est disponible"
    )
    model_name: str = Field(..., min_length=1, description="Nom du fichier de poids servi")
    model_path: str = Field(..., min_length=1, description="Chemin du modèle servi")
    model_version: Optional[str] = Field(
        default=None, description="Version logique du modèle promu"
    )
    run_id: Optional[str] = Field(
        default=None, description="Identifiant du run ayant produit le modèle"
    )
    dataset_version: Optional[str] = Field(default=None, description="Version du dataset utilisé")
    trained_at: Optional[str] = Field(
        default=None, description="Horodatage d'entraînement ou d'enregistrement"
    )
    metrics_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Résumé des métriques associées au modèle promu",
    )
    message: Optional[str] = Field(
        default=None, description="Message explicite si le manifeste est absent"
    )


class ErrorResponse(BaseModel):
    """Réponse en cas d'erreur."""

    error: str = Field(..., description="Message d'erreur")
    status_code: int = Field(default=500, description="Code HTTP de l'erreur")
