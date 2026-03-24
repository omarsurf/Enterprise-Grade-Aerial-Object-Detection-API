---
name: deploy
description: Promeut le dernier modèle entraîné vers le slot de production et redémarre l'API. Utiliser après un entraînement réussi.
argument-hint: "[--run-id specific_run]"
allowed-tools: Bash, Read, Grep
---

# Skill: Déploiement du modèle

Tu vas promouvoir un modèle entraîné vers le slot de production.

## 1. Vérification de l'état actuel

```bash
cd "$CLAUDE_PROJECT_DIR"
cat artifacts/promoted/manifest.json 2>/dev/null || echo "Aucun modèle promu actuellement"
```

## 2. Identification du run à promouvoir

```bash
ls -la artifacts/ | grep -v promoted | grep -v datasets
```

Arguments passés: $ARGUMENTS

## 3. Comparaison des métriques

Compare les métriques du nouveau run avec le modèle actuellement promu:
- mAP50
- mAP50-95
- Precision
- Recall

## 4. Promotion

```bash
make promote-model
```

Ou avec un run spécifique:
```bash
./venv/bin/python scripts/promote_model.py --run-id $1
```

## 5. Test de smoke

```bash
make smoke-inference
```

## 6. Redémarrage de l'API (si en cours)

```bash
# Si Docker
make docker-stop && make docker-run

# Si local
# Relancer make run
```

## Vérification finale

```bash
curl -s http://localhost:8000/model-info | jq .
```
