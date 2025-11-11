# Hydraulic Diagnostics SaaS - Makefile
# Quick commands for Docker management

.PHONY: help build up down restart logs clean test migrate

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all containers
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs (use: make logs SERVICE=backend_fastapi)
ifdef SERVICE
	docker-compose logs -f $(SERVICE)
else
	docker-compose logs -f
endif

clean: ## Stop and remove all containers, networks, volumes
	docker-compose down -v
	docker system prune -f

clean-all: ## Complete cleanup (WARNING: removes all Docker data)
	@echo "⚠️  This will remove ALL Docker containers, images, and volumes!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose down -v
	docker system prune -af --volumes

# Development commands
dev: ## Start in development mode
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-build: ## Build and start in development mode
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Database commands
migrate: ## Run database migrations
	docker-compose exec backend_fastapi alembic upgrade head
	docker-compose exec backend_django python manage.py migrate

makemigrations: ## Create new migrations
	docker-compose exec backend_fastapi alembic revision --autogenerate -m "$(MSG)"
	docker-compose exec backend_django python manage.py makemigrations

db-shell: ## Open database shell
	docker-compose exec postgres psql -U user -d hydraulic_db

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

# Testing
test: ## Run tests
	docker-compose exec backend_fastapi pytest
	docker-compose exec backend_django python manage.py test

# Service-specific commands
backend-shell: ## Open FastAPI backend shell
	docker-compose exec backend_fastapi bash

django-shell: ## Open Django shell
	docker-compose exec backend_django python manage.py shell

gnn-shell: ## Open GNN service shell
	docker-compose exec gnn_service bash

# Monitoring
ps: ## Show running containers
	docker-compose ps

stats: ## Show container stats
	docker stats

health: ## Check health status
	@echo "Backend FastAPI:"; curl -s http://localhost:8100/health/ | jq
	@echo "\nDjango Admin:"; curl -s http://localhost:8000/health/
	@echo "\nGNN Service:"; curl -s http://localhost:8001/gnn/health | jq

# Backup
backup-db: ## Backup PostgreSQL database
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U user hydraulic_db > backups/db_backup_$$(date +%Y%m%d_%H%M%S).sql

restore-db: ## Restore database (use: make restore-db FILE=backup.sql)
ifdef FILE
	docker-compose exec -T postgres psql -U user hydraulic_db < $(FILE)
else
	@echo "Error: specify FILE=path/to/backup.sql"
endif
