---
paths:
  - "scripts/*.py"
  - "artifacts/**/*"
  - "configs/*.yaml"
---

# Règles pour le Workflow ML

## Scripts d'entraînement
- Toujours utiliser les configs YAML, ne pas hardcoder les paramètres
- Chaque run doit générer un bundle complet: manifest.json, metrics.json, params.json
- Ne jamais modifier directement les fichiers dans `artifacts/promoted/`

## Artifacts
- Les manifests doivent être valides JSON avec les champs requis
- Utiliser `make validate-artifacts` après toute modification
- Le run_id doit être unique et descriptif

## Promotion de modèle
- Toujours comparer les métriques avant promotion
- La promotion copie le bundle ET le fichier .pt
- Vérifier avec `make smoke-inference` après promotion

## Configs YAML
- `data.yaml`: dataset paths et splits
- `train.yaml`: hyperparamètres d'entraînement
- `inference.yaml`: paramètres de tiling et seuils

## Métriques à surveiller
- mAP50 > 0.85 pour promotion
- mAP50-95 > 0.55 recommandé
- Precision et Recall équilibrés
