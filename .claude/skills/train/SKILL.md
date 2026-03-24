---
name: train
description: Lance un cycle complet d'entraînement du modèle YOLO OBB. Utiliser pour entraîner un nouveau modèle.
argument-hint: "[--run-id custom_name]"
allowed-tools: Bash, Read, Grep
---

# Skill: Entraînement du modèle YOLO OBB

Tu vas lancer un cycle complet d'entraînement. Suis ces étapes dans l'ordre:

## 1. Vérification préalable

```bash
cd "$CLAUDE_PROJECT_DIR"
make validate-artifacts
```

Vérifie que le dataset manifest existe et est valide.

## 2. Préparation des données (si nécessaire)

```bash
make prepare-data
```

## 3. Lancement de l'entraînement

```bash
make train
```

Arguments optionnels passés: $ARGUMENTS

## 4. Post-entraînement

Après l'entraînement:
1. Affiche le résumé des métriques depuis `artifacts/<run_id>/metrics.json`
2. Compare avec le modèle actuellement promu si disponible
3. Suggère la promotion si les métriques sont meilleures

## Notes importantes
- L'entraînement peut prendre plusieurs heures selon la config GPU
- Les artifacts sont automatiquement sauvegardés sous `artifacts/<run_id>/`
- Ne pas interrompre pendant l'écriture des checkpoints
