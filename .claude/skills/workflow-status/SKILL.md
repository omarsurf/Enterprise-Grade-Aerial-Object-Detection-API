---
name: workflow-status
description: Affiche l'état complet du workflow ML - datasets, runs, modèle promu, et santé de l'API
allowed-tools: Bash, Read, Grep, Glob
---

# Skill: État du Workflow ML

Tu vas analyser l'état complet du workflow ML.

## 1. Dataset

```bash
cd "$CLAUDE_PROJECT_DIR"
echo "=== DATASET MANIFEST ==="
cat artifacts/datasets/*/dataset_manifest.json 2>/dev/null | jq . || echo "Aucun dataset préparé"
```

## 2. Historique des runs

```bash
echo "=== RUNS D'ENTRAINEMENT ==="
for dir in artifacts/*/; do
    if [[ "$dir" != *"promoted"* ]] && [[ "$dir" != *"datasets"* ]]; then
        echo "--- $dir ---"
        cat "${dir}manifest.json" 2>/dev/null | jq '{run_id, training_date, metrics_summary}' || true
    fi
done
```

## 3. Modèle promu actuel

```bash
echo "=== MODÈLE PROMU ==="
cat artifacts/promoted/manifest.json 2>/dev/null | jq . || echo "Aucun modèle promu"
```

## 4. Fichier de poids

```bash
echo "=== POIDS DU MODÈLE ==="
ls -lh models/*.pt 2>/dev/null || echo "Aucun fichier .pt dans models/"
```

## 5. État de l'API

```bash
echo "=== SANTÉ API ==="
curl -s http://localhost:8000/health 2>/dev/null | jq . || echo "API non accessible"
```

## 6. Validation des artifacts

```bash
echo "=== VALIDATION ==="
make validate-artifacts 2>&1 || true
```

## Résumé

Génère un résumé clair:
- Est-ce que le dataset est prêt?
- Combien de runs d'entraînement?
- Quel est le modèle actuellement servi?
- L'API est-elle opérationnelle?
- Y a-t-il des problèmes à résoudre?
