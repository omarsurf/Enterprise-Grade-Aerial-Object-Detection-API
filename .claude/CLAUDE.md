# Aerial OBB Detection API - Instructions Claude Code

## Projet
API FastAPI pour détection OBB (Oriented Bounding Box) sur images aériennes DOTA avec workflow ML local reproductible.

## Commandes principales
- **Build/Install**: `make install` ou `make dev`
- **Tests**: `make test` ou `make test-cov`
- **Lint/Format**: `make lint` / `make format`
- **Serveur local**: `make run` (port 8000)
- **Validation artifacts**: `make validate-artifacts`

## Workflow ML
1. `make prepare-data` - Valide le dataset et génère manifeste
2. `make train` - Entraîne le modèle YOLO OBB (artifacts sous `artifacts/<run_id>/`)
3. `make evaluate` - Évalue le dernier run
4. `make promote-model` - Promeut vers `models/best_tiled.pt`
5. `make smoke-inference` - Test de fumée

## Structure du code
- `app/` - API FastAPI (main.py, inference.py, schemas.py, artifacts.py)
- `scripts/` - Scripts workflow ML (train_obb.py, evaluate_obb.py, promote_model.py)
- `configs/` - Configs YAML (data.yaml, train.yaml, inference.yaml)
- `artifacts/` - Manifestes et métriques des runs
- `models/` - Poids du modèle servi (best_tiled.pt)

## Conventions
- Code en Python 3.9+
- Formatter: ruff
- Docstrings en français
- Type hints obligatoires
- Tests dans `tests/` avec pytest

## Artifacts ML
- Chaque run génère: `manifest.json`, `metrics.json`, `params.json`
- Modèle promu: copie vers `artifacts/promoted/` + `models/best_tiled.pt`
- Dataset manifest: `artifacts/datasets/<id>/dataset_manifest.json`

## Règles importantes
- Ne jamais commit les fichiers .pt (poids modèle) ou images dans `data/`
- Toujours valider les artifacts après modification: `make validate-artifacts`
- L'API démarre en mode dégradé si le modèle n'est pas trouvé
- Pre-commit hooks actifs: ruff check + format

## Skills disponibles
- `/train` - Lance un cycle complet d'entraînement
- `/deploy` - Promeut et déploie le modèle
- `/test-api` - Teste l'API avec des images de test
- `/workflow-status` - Vérifie l'état du workflow ML
