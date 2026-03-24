#!/usr/bin/env python3
"""
Promotes a validated run to the canonical best_tiled.pt serving slot.
"""

import argparse
from pathlib import Path

from common import resolve_repo_path

from app.artifacts import promote_run_artifacts, read_latest_run_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a run bundle to the served model slot.")
    parser.add_argument(
        "--run-id", default=None, help="Run id to promote. Defaults to the latest run."
    )
    parser.add_argument(
        "--source-model-path",
        default=None,
        help="Optional explicit weights file. Defaults to source_model_path from the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or read_latest_run_id()
    if not run_id:
        raise ValueError("No run id provided and no latest run id recorded yet.")

    source_model_path = None
    if args.source_model_path:
        source_model_path = resolve_repo_path(args.source_model_path)

    manifest = promote_run_artifacts(
        run_id=run_id,
        source_model_path=Path(source_model_path) if source_model_path else None,
    )

    print(f"[OK] Promoted run: {run_id}")
    print(f"[OK] Served model: {manifest['model_path']}")


if __name__ == "__main__":
    main()
