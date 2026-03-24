---
name: test-api
description: Teste l'API avec des images de test et affiche les résultats. Utiliser pour vérifier que l'API fonctionne correctement.
argument-hint: "[image_path]"
allowed-tools: Bash, Read, Grep, Glob
---

# Skill: Test de l'API

Tu vas tester l'API de détection OBB.

## 1. Vérification que l'API tourne

```bash
curl -s http://localhost:8000/health | jq .
```

## 2. Informations du modèle

```bash
curl -s http://localhost:8000/model-info | jq .
```

## 3. Test de prédiction

Image spécifiée: $ARGUMENTS

Si aucune image spécifiée, utilise une image de test:

```bash
cd "$CLAUDE_PROJECT_DIR"
# Liste les images de test disponibles
ls data/inference_test/*.png 2>/dev/null || ls data/raw/images/*.png 2>/dev/null | head -5
```

### Test avec une image

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@$1" | jq .
```

### Test avec sauvegarde

```bash
curl -X POST "http://localhost:8000/predict-and-save" \
  -F "file=@$1" | jq .
```

## 4. Analyse des résultats

Après le test:
1. Affiche le nombre de détections
2. Liste les classes détectées
3. Montre les scores de confiance
4. Indique le chemin de l'image annotée (si predict-and-save)

## 5. Tests automatisés

```bash
make test
```
