# Aerial Object Detection API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/YOLO-v11_OBB-00FFFF?style=for-the-badge" alt="YOLO"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<p align="center">
  <strong>Production-ready REST API for Oriented Bounding Box (OBB) detection on aerial/satellite imagery</strong>
</p>

---

## Why OBB Detection?

Traditional bounding boxes fail on rotated objects. **Oriented Bounding Boxes** provide precise detection for:
- Aircraft at any angle on runways
- Ships in harbors with varying orientations
- Vehicles on roads regardless of direction

```
Traditional Box (Imprecise)          Oriented Box (Precise)
┌─────────────────┐                      ╱╲
│    ✈️            │                    ╱  ╲
│                 │         vs        ╱ ✈️  ╲
│                 │                   ╲    ╱
└─────────────────┘                    ╲  ╱
                                        ╲╱
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Oriented Detection** | Rotated bounding boxes with 4-point polygons |
| **High-Resolution Tiling** | Process 4000x6000+ images via smart overlap |
| **Multi-Class** | Planes, ships, small & large vehicles |
| **REST API** | FastAPI with auto-generated Swagger docs |
| **Reproducible ML** | Full training pipeline with artifact tracking |
| **Production Ready** | Docker, health checks, model versioning |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            CLIENT REQUEST                               │
│                       POST /predict + image.png                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI SERVER                                │
│                                                                         │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────────────┐ │
│  │  Validate  │───▶│   Decode   │───▶│      TILED INFERENCE           │ │
│  │  Request   │    │   Image    │    │                                │ │
│  └────────────┘    └────────────┘    │   ┌──────┬──────┬──────┐       │ │
│                                      │   │ T1   │ T2   │ T3   │       │ │
│                                      │   ├──────┼──────┼──────┤       │ │
│                                      │   │ T4   │ T5   │ T6   │       │ │
│                                      │   ├──────┼──────┼──────┤       │ │
│                                      │   │ T7   │ T8   │ T9   │       │ │
│                                      │   └──────┴──────┴──────┘       │ │
│                                      │                                │ │
│                                      │   YOLO v11 OBB + NMS Merge     │ │
│                                      └────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           JSON RESPONSE                                 │
│                                                                         │
│   {                                                                     │
│     "image_width": 3875,                                                │
│     "image_height": 5502,                                               │
│     "nb_detections": 20,                                                │
│     "detections": [                                                     │
│       {                                                                 │
│         "class_name": "plane",                                          │
│         "confidence": 0.94,                                             │
│         "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]                 │
│       }                                                                 │
│     ]                                                                   │
│   }                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

```
┌────────────────────────────────────────────────────────┐
│                   MODEL METRICS                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│   mAP@50      ████████████████████░░░░  88.5%         │
│   mAP@50-95   ███████████░░░░░░░░░░░░░  57.4%         │
│   Precision   █████████████████░░░░░░░  87.3%         │
│   Recall      ████████████████░░░░░░░░  83.5%         │
│                                                        │
├────────────────────────────────────────────────────────┤
│   Classes: plane | ship | small-vehicle | large-vehicle│
│   Architecture: YOLOv11n-OBB                           │
│   Tile Size: 1024px | Overlap: 200px                   │
└────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install the CPU-only API environment

```bash
git clone <repository>
cd aerial-obb-api
make install
```

Equivalent manual install:

```bash
pip install -r requirements/api-cpu.txt
pip install -e . --no-deps
```

The supported runtime flow is driven by `requirements/*.txt`. Running `pip install -e .` alone is not a supported way to choose the CPU/GPU backend.

### 2. Run the API

```bash
make run
# Server at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 3. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@satellite_image.png"
```

### 4. Docker Deployment

```bash
docker build -t aerial-obb-api .
docker run -p 8000:8000 aerial-obb-api
```

The official container image is CPU-only and intended for API inference, not training.

### 5. Development and training environments

```bash
make dev       # CPU-only dev/test environment
make train-env # Local training environment
```

`make train-env` keeps training dependencies separate from the API image and may resolve GPU-capable packages depending on the local machine.

### 6. CI profile

GitHub Actions is intentionally CPU-only for lint, tests, artifact validation, and Docker verification.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/model-info` | GET | Model metadata & metrics |
| `/predict` | POST | OBB detection → JSON |
| `/predict-and-save` | POST | Detection → JSON + annotated image |

### Example Response

```json
{
  "image_width": 1156,
  "image_height": 1483,
  "nb_detections": 45,
  "detections": [
    {
      "class_id": 2,
      "class_name": "small-vehicle",
      "confidence": 0.79,
      "polygon": [
        [246.12, 315.31],
        [261.64, 298.95],
        [239.18, 277.64],
        [223.66, 294.00]
      ]
    }
  ]
}
```

---

## ML Workflow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   PREPARE    │───▶│    TRAIN     │───▶│   EVALUATE   │───▶│   PROMOTE    │
│    DATA      │    │    MODEL     │    │   METRICS    │    │   TO PROD    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  dataset_          artifacts/            metrics.json      models/best.pt
  manifest.json     <run_id>/                              artifacts/promoted/
```

```bash
make train-env       # Install the local training environment
make prepare-data    # Validate dataset structure
make train           # Train YOLO OBB model
make evaluate        # Compute validation metrics
make promote-model   # Deploy to production slot
make smoke-inference # End-to-end API test
```

---

## Project Structure

```
aerial-obb-api/
├── app/                      # FastAPI application
│   ├── main.py              # API endpoints
│   ├── inference.py         # Tiled OBB detection engine
│   ├── schemas.py           # Pydantic request/response models
│   ├── config.py            # Configuration management
│   └── artifacts.py         # ML artifact helpers
├── models/                   # Production model weights
│   ├── best_tiled.pt        # Served model (5.5MB)
│   └── best_baseline.pt     # Comparison baseline
├── scripts/                  # ML workflow automation
│   ├── train_obb.py         # Training script
│   ├── evaluate_obb.py      # Evaluation script
│   └── promote_model.py     # Model promotion
├── configs/                  # YAML configurations
│   ├── data.yaml            # Dataset paths
│   ├── train.yaml           # Training hyperparameters
│   └── inference.yaml       # Inference settings
├── requirements/             # CPU API, dev, lint, and training dependency sets
├── artifacts/                # ML run tracking
│   ├── promoted/            # Production model metadata
│   └── <run_id>/            # Per-run manifests & metrics
├── tests/                    # Test suite
├── Dockerfile               # Container deployment
├── Makefile                 # Development commands
└── MODEL_CARD.md            # Model documentation
```

---

## Technical Stack

| Category | Technology |
|----------|------------|
| **API Framework** | FastAPI + Uvicorn |
| **ML Model** | Ultralytics YOLOv11-OBB |
| **Validation** | Pydantic v2 |
| **Image Processing** | OpenCV, NumPy |
| **Containerization** | Docker |
| **Code Quality** | Ruff |
| **Testing** | Pytest |

---

## Use Cases

- **Airport Monitoring**: Aircraft detection and counting
- **Maritime Surveillance**: Ship tracking in ports
- **Traffic Analysis**: Vehicle detection in parking/roads
- **Urban Planning**: Infrastructure assessment
- **Defense & Security**: Aerial reconnaissance

---

## Improving Small Vehicle Detection

The current model achieves **88.5% mAP@50** across all classes. For **small vehicles specifically**, detection can be improved with these strategies:

### Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│              SMALL VEHICLE DETECTION IMPROVEMENT ROADMAP                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CURRENT          QUICK WINS           ADVANCED            TARGET       │
│  ────────         ──────────           ────────            ──────       │
│                                                                         │
│  mAP: 88%    ──▶  Smaller Tiles   ──▶  Model Upgrade  ──▶  mAP: 93%+   │
│                   + More Overlap       + Data Aug                       │
│                                                                         │
│  Tile: 1024  ──▶  Tile: 640       ──▶  YOLOv11s-OBB   ──▶  Tile: 512   │
│  Overlap: 200    Overlap: 300         + Mosaic Aug        Overlap: 256  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Quick Wins (No Retraining)

Modify `configs/inference.yaml`:

```yaml
inference:
  tile_size: 640        # Smaller tiles = larger objects relative to tile
  overlap: 300          # More overlap = fewer missed edge detections
  confidence_threshold: 0.45  # Lower threshold for small objects
```

**Expected improvement**: +2-4% mAP on small vehicles

### 2. Training Improvements

Modify `configs/train.yaml`:

```yaml
training:
  imgsz: 1280           # Higher resolution training
  mosaic: 1.0           # Enable mosaic augmentation
  scale: 0.9            # More scale variation
  epochs: 100           # More training epochs

  # Small object focus
  box: 7.5              # Increase box loss weight
  cls: 0.5              # Standard classification loss
```

### 3. Model Architecture Upgrade

```bash
# Current: YOLOv11n-OBB (nano) - 5.5MB
# Upgrade: YOLOv11s-OBB (small) - 18MB
# Premium: YOLOv11m-OBB (medium) - 42MB

# In configs/train.yaml:
run:
  initial_weights: yolo11s-obb.pt  # or yolo11m-obb.pt
```

| Model | Size | mAP@50 (expected) | Inference Time |
|-------|------|-------------------|----------------|
| YOLOv11n-OBB | 5.5MB | 88.5% | 15ms |
| YOLOv11s-OBB | 18MB | 91-92% | 25ms |
| YOLOv11m-OBB | 42MB | 93-94% | 40ms |

### 4. Data Augmentation for Small Objects

```yaml
# Advanced augmentations in train.yaml
training:
  hsv_h: 0.015          # Hue variation
  hsv_s: 0.7            # Saturation variation
  hsv_v: 0.4            # Value variation
  flipud: 0.5           # Vertical flip (aerial views)
  fliplr: 0.5           # Horizontal flip
  mixup: 0.15           # MixUp augmentation
  copy_paste: 0.3       # Copy-paste small objects
```

### 5. Recommended Configuration for Maximum Accuracy

```yaml
# configs/train_high_accuracy.yaml
run:
  initial_weights: yolo11s-obb.pt
  prefix: high_accuracy

training:
  epochs: 150
  imgsz: 1280
  batch: 8
  mosaic: 1.0
  scale: 0.9
  copy_paste: 0.3

inference:
  tile_size: 512
  overlap: 256
  confidence_threshold: 0.4
```

**Expected results**: mAP@50 > 92% on small vehicles

---

## Development

```bash
make install             # Install the CPU-only API environment
make dev                 # Install the CPU-only dev/test environment
make train-env           # Install the local training environment
make test                # Run test suite
make check               # Lint, tests, and artifact validation
make smoke-inference     # API smoke test
make validate-artifacts  # Validate ML artifacts
```

Omar PIRO - Ml engineer
