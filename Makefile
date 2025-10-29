SHELL := /bin/bash

# Default env file
ENV ?= .env

.PHONY: help
help:
	@echo "Useful targets:"
	@echo "  make up                - start dev stack (docker-compose.yml)"
	@echo "  make down              - stop dev stack"
	@echo "  make logs              - follow dev logs"
	@echo "  make migrate           - run Django migrations"
	@echo "  make superuser         - create Django superuser"
	@echo "  make up-prod           - start prod stack (docker-compose.prod.yml)"
	@echo "  make down-prod         - stop prod stack"
	@echo "  make logs-prod         - follow prod logs"
	@echo "  make test              - run tests inside dev backend"
	@echo "  make lint              - run ruff lint"
	@echo "  make fmt               - run ruff format"

# -------- Dev stack --------
up:
	docker compose --env-file $(ENV) up --build -d

down:
	docker compose --env-file $(ENV) down

logs:
	docker compose --env-file $(ENV) logs -f --tail=200

migrate:
	docker compose --env-file $(ENV) exec backend python backend/manage.py migrate

superuser:
	docker compose --env-file $(ENV) exec backend python backend/manage.py createsuperuser

test:
	docker compose --env-file $(ENV) exec backend pytest -q

lint:
	docker compose --env-file $(ENV) exec backend ruff check .

fmt:
	docker compose --env-file $(ENV) exec backend ruff format .

# -------- Prod stack --------
up-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) up --build -d

down-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) down

logs-prod:
	docker compose -f docker-compose.prod.yml --env-file $(ENV) logs -f --tail=200
