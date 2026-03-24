---
paths:
  - "app/**/*.py"
  - "tests/**/*.py"
---

# Règles pour le développement API

## FastAPI
- Utiliser les type hints Pydantic pour tous les endpoints
- Documenter avec docstrings en français
- Gérer les erreurs avec HTTPException appropriées
- Codes de statut: 200 OK, 400 Bad Request, 413 Too Large, 503 Service Unavailable

## Inférence
- L'inférence est thread-safe via ThreadPoolExecutor
- Le détecteur a un lock pour sérialiser l'accès au modèle
- Ne pas bloquer l'event loop asyncio avec des opérations synchrones

## Tests
- Chaque endpoint doit avoir des tests
- Utiliser pytest fixtures pour le client TestClient
- Mocker le modèle pour les tests rapides: `--mock` flag

## Schemas
- Définir dans `app/schemas.py`
- Utiliser Pydantic BaseModel
- Inclure des exemples dans la doc OpenAPI

## Gestion des fichiers
- Valider extension et MIME type
- Limite de taille: MAX_FILE_SIZE_MB
- Nettoyer les ressources après traitement
