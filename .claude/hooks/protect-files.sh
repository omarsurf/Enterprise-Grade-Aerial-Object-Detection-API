#!/bin/bash
# Hook pour protéger les fichiers sensibles de modification accidentelle

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Patterns de fichiers protégés
PROTECTED_PATTERNS=(
    ".env"
    ".git/"
    "models/*.pt"
    "data/raw/"
    "data/tiled/"
    "artifacts/promoted/manifest.json"
    "venv/"
    "__pycache__"
)

for pattern in "${PROTECTED_PATTERNS[@]}"; do
    if [[ "$FILE_PATH" == *"$pattern"* ]]; then
        echo "Bloqué: '$FILE_PATH' correspond au pattern protégé '$pattern'" >&2
        echo "Utilisez les scripts appropriés pour modifier ces fichiers (make promote-model, etc.)" >&2
        exit 2
    fi
done

# Avertissement pour les fichiers de config sensibles
WARNING_PATTERNS=(
    "configs/"
    "requirements"
    "pyproject.toml"
)

for pattern in "${WARNING_PATTERNS[@]}"; do
    if [[ "$FILE_PATH" == *"$pattern"* ]]; then
        echo "Note: Modification d'un fichier de configuration important: $FILE_PATH" >&2
    fi
done

exit 0
