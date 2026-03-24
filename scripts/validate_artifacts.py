#!/usr/bin/env python3
"""
Validates local dataset/model manifests and the promoted metadata bundle.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from common import ROOT_DIR

from app.artifacts import (  # noqa: E402
    ARTIFACTS_DIR,
    MODEL_PATH,
    PROMOTED_MANIFEST_PATH,
    PROMOTED_METRICS_PATH,
    PROMOTED_PARAMS_PATH,
    REQUIRED_DATASET_MANIFEST_KEYS,
    REQUIRED_MODEL_MANIFEST_KEYS,
    read_json,
    validate_required_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Validate local dataset/model manifest structure under {ROOT_DIR}."
    )
    parser.add_argument(
        "--require-promoted-bundle",
        action="store_true",
        help="Fail when the promoted manifest/params/metrics bundle is missing.",
    )
    return parser.parse_args()


def collect_model_manifest_paths() -> List[Path]:
    paths = []
    for manifest_path in sorted(ARTIFACTS_DIR.glob("*/manifest.json")):
        if manifest_path.parent.name in {"datasets", "promoted"}:
            continue
        paths.append(manifest_path)
    return paths


def resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def main() -> None:
    args = parse_args()
    errors = []

    dataset_manifests = sorted(ARTIFACTS_DIR.glob("datasets/*/dataset_manifest.json"))
    for manifest_path in dataset_manifests:
        payload = read_json(manifest_path)
        missing_keys = validate_required_keys(payload, REQUIRED_DATASET_MANIFEST_KEYS)
        if missing_keys:
            errors.append(f"{manifest_path}: missing dataset keys {missing_keys}")

    model_manifests = collect_model_manifest_paths()
    for manifest_path in model_manifests:
        payload = read_json(manifest_path)
        missing_keys = validate_required_keys(payload, REQUIRED_MODEL_MANIFEST_KEYS)
        if missing_keys:
            errors.append(f"{manifest_path}: missing model keys {missing_keys}")
            continue

        dataset_manifest_path = payload.get("dataset_manifest_path")
        if dataset_manifest_path and not resolve_manifest_path(dataset_manifest_path).exists():
            errors.append(
                f"{manifest_path}: dataset_manifest_path does not exist -> {dataset_manifest_path}"
            )

        source_model_path = payload.get("source_model_path")
        if source_model_path and not resolve_manifest_path(source_model_path).exists():
            errors.append(
                f"{manifest_path}: source_model_path does not exist -> {source_model_path}"
            )

    promoted_bundle_exists = (
        PROMOTED_MANIFEST_PATH.exists()
        and PROMOTED_METRICS_PATH.exists()
        and PROMOTED_PARAMS_PATH.exists()
    )
    if args.require_promoted_bundle and not promoted_bundle_exists:
        errors.append("Promoted bundle is missing under artifacts/promoted/")

    if promoted_bundle_exists:
        promoted_manifest = read_json(PROMOTED_MANIFEST_PATH)
        promoted_missing = validate_required_keys(promoted_manifest, REQUIRED_MODEL_MANIFEST_KEYS)
        if promoted_missing:
            errors.append(f"{PROMOTED_MANIFEST_PATH}: missing promoted keys {promoted_missing}")
        dataset_manifest_path = promoted_manifest.get("dataset_manifest_path")
        if dataset_manifest_path and not resolve_manifest_path(dataset_manifest_path).exists():
            errors.append(
                f"{PROMOTED_MANIFEST_PATH}: dataset_manifest_path does not exist -> {dataset_manifest_path}"
            )
        run_id = promoted_manifest.get("run_id")
        if run_id:
            run_manifest_path = ARTIFACTS_DIR / run_id / "manifest.json"
            if not run_manifest_path.exists():
                errors.append(
                    f"Promoted manifest points to missing run bundle: {run_manifest_path}"
                )
        if promoted_manifest.get("model_name") != MODEL_PATH.name:
            errors.append(
                f"Promoted model_name should match {MODEL_PATH.name}, got {promoted_manifest.get('model_name')}"
            )

    if errors:
        print("[ERROR] Artifact validation failed:")
        for error in errors:
            print(f" - {error}")
        sys.exit(1)

    print(f"[OK] Validated {len(dataset_manifests)} dataset manifests")
    print(f"[OK] Validated {len(model_manifests)} model manifests")
    if promoted_bundle_exists:
        print(f"[OK] Promoted bundle present for {MODEL_PATH.name}")
    else:
        print("[OK] No promoted bundle present")


if __name__ == "__main__":
    main()
