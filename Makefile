.PHONY: install dev test test-cov lint format run prepare-data train evaluate promote-model smoke-inference validate-artifacts docker-build docker-run docker-stop clean help

# Variables
PYTHON := python3
VENV := venv
PORT := 8000
DATA_CONFIG := configs/data.yaml
TRAIN_CONFIG := configs/train.yaml
INFERENCE_CONFIG := configs/inference.yaml
SMOKE_FLAGS ?=

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

dev: ## Install development dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

test: ## Run tests
	$(VENV)/bin/pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(VENV)/bin/pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=html

lint: ## Run linter
	$(VENV)/bin/ruff check app/ tests/

format: ## Format code
	$(VENV)/bin/ruff format app/ tests/
	$(VENV)/bin/ruff check --fix app/ tests/

run: ## Run the API locally
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

prepare-data: ## Validate dataset layout and emit a dataset manifest
	$(VENV)/bin/python scripts/prepare_data.py --config $(DATA_CONFIG)

train: ## Train the tiled YOLO OBB model and record artifacts
	$(VENV)/bin/python scripts/train_obb.py --config $(TRAIN_CONFIG) --data-config $(DATA_CONFIG) --inference-config $(INFERENCE_CONFIG)

evaluate: ## Evaluate the latest run and refresh its metrics bundle
	$(VENV)/bin/python scripts/evaluate_obb.py --data-config $(DATA_CONFIG)

promote-model: ## Promote the latest validated run to models/best_tiled.pt
	$(VENV)/bin/python scripts/promote_model.py

smoke-inference: ## Run a smoke inference against the promoted API surface
	$(VENV)/bin/python scripts/smoke_inference.py --config $(INFERENCE_CONFIG) $(SMOKE_FLAGS)

validate-artifacts: ## Validate dataset/model manifests and promoted metadata
	$(VENV)/bin/python scripts/validate_artifacts.py --require-promoted-bundle

docker-build: ## Build Docker image
	docker build -t aerial-obb-api:latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-logs: ## Show Docker container logs
	docker-compose logs -f

docker-test: ## Run tests inside Docker container
	docker run --rm aerial-obb-api:latest pytest tests/ -v

clean: ## Clean up generated files
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf outputs/predictions/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

check: lint test validate-artifacts ## Run linter, tests, and artifact validation
