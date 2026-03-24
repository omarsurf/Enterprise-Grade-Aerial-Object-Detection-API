"""
Shared helpers for local workflow scripts.
"""

import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml  # noqa: E402


def resolve_repo_path(path_value: str) -> Path:
    """Resolves a repo-relative path from a config value."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Loads a YAML config file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def write_yaml_config(path: Path, payload: Dict[str, Any]) -> None:
    """Writes a YAML config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
