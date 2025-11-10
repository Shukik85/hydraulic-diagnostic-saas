# Hydraulic ML Service - Container Operations
# Use GPU by default, CPU as fallback

.PHONY: help build train serve test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build ML service (GPU-capable)
	docker compose build ml-service

build-cpu: ## Build ML service (CPU-only)
	docker compose build ml-service-cpu

build-all: ## Build all variants
	docker compose build

# Training targets  
train: ## Train models (GPU full search)
	docker compose --profile training --profile gpu run --rm ml-trainer

train-cpu: ## Train models (CPU fast)
	docker compose --profile training --profile cpu run --rm ml-trainer-cpu

train-only: ## Train single model (usage: make train-only MODEL=xgboost)
	@if [ -z "$(MODEL)" ]; then echo "Usage: make train-only MODEL=catboost|xgboost|random_forest|adaptive"; exit 1; fi
	docker compose --profile training --profile gpu run --rm ml-trainer python scripts/train/train_production.py --gpu --only $(MODEL)

train-dataset: ## Build dataset from RAW
	docker compose --profile training run --rm ml-trainer python make_uci_dataset.py

# ONNX Export targets
onnx-export: ## Export all models to ONNX format
	@echo "Exporting models to ONNX..."
	python scripts/onnx/export_to_onnx.py --models-dir ./models --output-dir ./models/onnx

onnx-export-docker: ## Export models to ONNX inside Docker
	docker compose --profile inference run --rm ml-service python scripts/onnx/export_to_onnx.py --models-dir ./models --output-dir ./models/onnx

onnx-validate: ## Validate ONNX models
	@echo "Validating ONNX models..."
	@if [ -f "./models/onnx/onnx_export_report.json" ]; then cat ./models/onnx/onnx_export_report.json | jq; else echo "No ONNX export report found. Run 'make onnx-export' first."; fi

# Inference targets
serve: ## Start ML inference service (GPU)
	docker compose --profile inference --profile gpu up -d ml-service

serve-cpu: ## Start ML inference service (CPU)
	docker compose --profile inference --profile cpu up -d ml-service-cpu

serve-onnx: ## Start ONNX optimized service (ultra-fast)
	@echo "Starting ONNX optimized service on port 8002..."
	python onnx_predict.py

serve-onnx-docker: ## Start ONNX service in Docker
	docker compose --profile inference run --rm -p 8002:8002 ml-service python onnx_predict.py

stop: ## Stop ML service
	docker compose --profile inference down

# Testing targets
test-predict: ## Test prediction endpoint
	@echo "Testing prediction endpoint..."
	curl -X POST http://localhost:8001/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [150.0, 65.0, 42.0, 89.0, 12.0, 78.0, 34.0, 56.0, 23.0, 91.0, 67.0, 45.0, 88.0, 29.0, 73.0, 51.0, 94.0, 37.0, 62.0, 19.0, 85.0, 48.0, 76.0, 33.0, 59.0]}' | jq

test-onnx: ## Test ONNX optimized endpoint
	@echo "Testing ONNX endpoint (port 8002)..."
	curl -X POST http://localhost:8002/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [150.0, 65.0, 42.0, 89.0, 12.0, 78.0, 34.0, 56.0, 23.0, 91.0, 67.0, 45.0, 88.0, 29.0, 73.0, 51.0, 94.0, 37.0, 62.0, 19.0, 85.0, 48.0, 76.0, 33.0, 59.0]}' | jq

test-onnx-fast: ## Test ONNX fast endpoint (CatBoost only)
	@echo "Testing ONNX fast endpoint..."
	curl -X POST http://localhost:8002/predict/fast \
		-H "Content-Type: application/json" \
		-d '{"features": [150.0, 65.0, 42.0, 89.0, 12.0, 78.0, 34.0, 56.0, 23.0, 91.0, 67.0, 45.0, 88.0, 29.0, 73.0, 51.0, 94.0, 37.0, 62.0, 19.0, 85.0, 48.0, 76.0, 33.0, 59.0]}' | jq

test-health: ## Test health endpoints
	@echo "Testing health endpoints..."
	curl http://localhost:8001/healthz | jq
	curl http://localhost:8001/readyz | jq
	curl http://localhost:8001/models | jq

test-metrics: ## Show Prometheus metrics
	curl http://localhost:8001/metrics

benchmark-onnx: ## Benchmark ONNX vs native inference
	@echo "Running benchmark: ONNX vs Native..."
	python scripts/onnx/benchmark_onnx.py

# Utility targets
logs: ## Show ML service logs
	docker compose --profile inference logs -f ml-service

logs-train: ## Show training logs
	docker compose --profile training logs ml-trainer

shell: ## Open shell in ML container
	docker compose --profile inference run --rm ml-service bash

clean: ## Clean containers and images
	docker compose --profile training --profile inference down -v
	docker system prune -f

# Development targets
dev-serve: ## Run inference locally (no container)
	python simple_predict.py

dev-serve-onnx: ## Run ONNX inference locally
	python onnx_predict.py

dev-train: ## Run training locally (no container)
	python scripts/train/train_production.py --gpu

dev-test: ## Test local service
	@echo "Testing local service at http://localhost:8001"
	curl http://localhost:8001/healthz | jq

# E2E targets
e2e: build train serve test-predict ## Full E2E: build -> train -> serve -> test

e2e-onnx: build train onnx-export serve-onnx test-onnx ## Full E2E with ONNX optimization

# Production deployment helpers
deploy-models: ## Deploy models to production (copy to volume)
	@echo "Copying models to production volume..."
	docker compose --profile inference exec ml-service ls -la /app/models/

backup-models: ## Backup current models
	mkdir -p backups/models-$$(date +%Y%m%d-%H%M%S)
	cp -r models/* backups/models-$$(date +%Y%m%d-%H%M%S)/
	@echo "Models backed up to backups/models-$$(date +%Y%m%d-%H%M%S)/"

# Default target
default: help
