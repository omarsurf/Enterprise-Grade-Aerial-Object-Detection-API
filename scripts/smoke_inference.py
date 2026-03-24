#!/usr/bin/env python3
from __future__ import annotations

"""
Runs a lightweight smoke inference against the promoted API surface.
"""
import argparse
from unittest.mock import patch

from common import load_yaml_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight inference smoke test.")
    parser.add_argument("--config", default="configs/inference.yaml", help="Inference config YAML.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a mocked detector instead of loading the actual promoted model.",
    )
    return parser.parse_args()


def build_sample_image(config: dict) -> np.ndarray:
    import numpy as np  # noqa: E402

    smoke_cfg = config["smoke_test"]
    height = int(smoke_cfg["image_height"])
    width = int(smoke_cfg["image_width"])
    color = tuple(int(channel) for channel in smoke_cfg["background_bgr"])
    image = np.full((height, width, 3), color, dtype=np.uint8)

    import cv2  # noqa: E402

    cv2.rectangle(image, (250, 250), (500, 500), (180, 180, 180), -1)
    return image


def run_mock_smoke_test(image: np.ndarray) -> None:
    import cv2  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402

    from app.main import app  # noqa: E402
    from app.schemas import Detection, ModelInfoResponse, PredictionResponse  # noqa: E402

    mocked_prediction = PredictionResponse(
        image_width=image.shape[1],
        image_height=image.shape[0],
        nb_detections=1,
        detections=[
            Detection(
                class_id=0,
                class_name="plane",
                confidence=0.99,
                polygon=[[250, 250], [500, 250], [500, 500], [250, 500]],
            )
        ],
    )

    model_info = ModelInfoResponse(
        model_loaded=True,
        manifest_available=True,
        model_name="best_tiled.pt",
        model_path="models/best_tiled.pt",
        model_version="smoke-test",
        run_id="smoke-test",
        dataset_version="local-v1",
        trained_at="2026-03-23T00:00:00+00:00",
        metrics_summary={"mAP50-95": 0.5},
        message=None,
    )

    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Unable to encode the smoke test image.")

    with (
        patch("app.main.detector") as mock_detector,
        patch(
            "app.main.get_runtime_model_info",
            return_value=model_info,
        ),
    ):
        mock_detector.is_loaded.return_value = True
        mock_detector.predict.return_value = mocked_prediction
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={"file": ("smoke.png", buffer.tobytes(), "image/png")},
        )
        assert response.status_code == 200, response.text

        info_response = client.get("/model-info")
        assert info_response.status_code == 200, info_response.text
        assert info_response.json()["manifest_available"] is True

    print("[OK] Mock smoke inference succeeded")


def run_real_smoke_test(image: np.ndarray) -> None:
    from app.artifacts import build_model_info  # noqa: E402
    from app.inference import detector  # noqa: E402

    if not detector.is_loaded():
        detector.load_model()

    prediction = detector.predict(image)
    if prediction.image_width != image.shape[1] or prediction.image_height != image.shape[0]:
        raise AssertionError("Prediction image dimensions do not match the generated input.")

    model_info = build_model_info(model_loaded=detector.is_loaded())
    print(
        f"[OK] Real smoke inference succeeded with {prediction.nb_detections} detections "
        f"on model {model_info['model_name']}"
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(resolve_repo_path(args.config))
    image = build_sample_image(config)

    if args.mock:
        run_mock_smoke_test(image)
    else:
        run_real_smoke_test(image)


if __name__ == "__main__":
    main()
