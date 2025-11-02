SHELL := /bin/bash

# Default env file
ENV ?= .env

.PHONY: help
help:
	@echo "🛠️  Hydraulic Diagnostic SaaS - Available Commands:"
	@echo ""
	@echo "📦 Main Development:"
	@echo "  make dev               - start dev stack (alias for 'up')"
	@echo "  make up                - start dev stack (docker-compose.dev.yml)"
	@echo "  make down              - stop dev stack"
	@echo "  make logs              - follow dev logs"
	@echo "  make init-data         - initialize demo data and RAG system"
	@echo ""
	@echo "🗄️  Database & Migrations:"
	@echo "  make migrate           - run Django migrations"
	@echo "  make superuser         - create Django superuser"
	@echo "  make shell             - open Django shell"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test              - run all tests"
	@echo "  make test-backend      - run backend tests only"
	@echo "  make test-rag          - run RAG system tests"
	@echo "  make smoke-test        - run smoke tests"
	@echo "  make lint              - run ruff lint"
	@echo "  make format            - run ruff format"
	@echo "  make check             - run pre-commit checks"
	@echo ""
	@echo "🚀 Production:"
	@echo "  make up-prod           - start prod stack (docker-compose.prod.yml)"
	@echo "  make down-prod         - stop prod stack"
	@echo "  make logs-prod         - follow prod logs"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make clean             - clean up containers and volumes"
	@echo "  make rebuild           - rebuild all containers"

# -------- Dev stack --------
dev: up

up:
	@if [ ! -f $(ENV) ]; then \
		echo "⚠️  .env file not found. Creating from .env.dev.example..."; \
		cp .env.dev.example $(ENV); \
	fi
	docker compose -f docker-compose.dev.yml --env-file $(ENV) up --build -d

down:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) down

logs:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) logs -f --tail=200

# -------- Database operations --------
migrate:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python manage.py migrate

superuser:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python manage.py createsuperuser

shell:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python manage.py shell

# -------- Data initialization --------
init-data:
	@echo "🔄 Initializing demo data and RAG system..."
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python manage.py generate_test_data
	@echo "✅ Demo data initialized successfully!"

# -------- Testing --------
test:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend pytest -v

test-backend:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend pytest backend/tests/ -v

test-rag:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python test_rag.py

smoke-test:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend python smoke_diagnostics.py

# -------- Code quality --------
lint:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend ruff check .

format:
	docker compose -f docker-compose.dev.yml --env-file $(ENV) exec backend ruff format .

check:
	@echo "🔍 Running pre-commit checks..."
	pre-commit run --all-files

# -------- Prod stack --------
up-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) up --build -d

down-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) down

logs-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) logs -f --tail=200

# -------- Utilities --------
clean:
	@echo "🧹 Cleaning up containers and volumes..."
	docker compose -f docker-compose.dev.yml down -v --remove-orphans
	docker system prune -f

rebuild:
	@echo "🔄 Rebuilding all containers..."
	docker compose -f docker-compose.dev.yml down
	docker compose -f docker-compose.dev.yml build --no-cache
	docker compose -f docker-compose.dev.yml up -d