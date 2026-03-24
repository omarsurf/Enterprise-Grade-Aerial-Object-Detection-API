#!/usr/bin/env python3
"""
Script to download sample aerial images for testing.
Uses public domain satellite imagery.
"""

import urllib.request
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

SAMPLE_IMAGES = [
    {
        "name": "sample_airport.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Sheremetyevo_Airport_2011.jpg/1280px-Sheremetyevo_Airport_2011.jpg",
        "description": "Airport aerial view (Wikimedia Commons, Public Domain)",
    }
]


def download_samples():
    """Download sample images for testing."""
    SAMPLES_DIR.mkdir(exist_ok=True)

    for sample in SAMPLE_IMAGES:
        output_path = SAMPLES_DIR / sample["name"]

        if output_path.exists():
            print(f"[SKIP] {sample['name']} already exists")
            continue

        print(f"[DOWNLOAD] {sample['name']}...")
        try:
            urllib.request.urlretrieve(sample["url"], output_path)
            print(f"[OK] Saved to {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {sample['name']}: {e}")


if __name__ == "__main__":
    download_samples()
