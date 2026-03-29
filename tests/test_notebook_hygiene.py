"""
Tests that keep notebooks lightweight and explicitly exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_stripped_and_repositioned():
    """Notebooks should stay lightweight and clearly secondary to Python modules."""
    notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))

    assert notebooks

    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        first_cell = notebook["cells"][0]
        all_sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        assert first_cell["cell_type"] == "markdown"
        assert "Exploratory only" in "".join(first_cell.get("source", []))
        assert "source of truth = python modules" in "".join(first_cell.get("source", [])).lower()
        assert "google.colab" not in all_sources
        assert "!pip install" not in all_sources

        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
