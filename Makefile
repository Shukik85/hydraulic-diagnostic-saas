# Hydraulic Diagnostic SaaS - Development Makefile
# Modern development workflow for 2025

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

# Project settings
PROJECT_NAME := hydraulic-diagnostic-saas
BACKEND_DIR := backend
FRONTEND_DIR := nuxt_frontend
DOCKER_COMPOSE_DEV := docker-compose.dev.yml
DOCKER_COMPOSE_PROD := docker-compose.prod.yml
PYTHON := python
PIP := pip
UV := uv
NODE := node
NPM := npm

# Check if commands exist
HAS_UV := $(shell command -v uv 2> /dev/null)
HAS_DOCKER := $(shell command -v docker 2> /dev/null)
HAS_DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null)

##@ 🚀 Development

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(CYAN)Installing development dependencies...$(RESET)"
	@if [ "$(HAS_UV)" ]; then \
		echo "$(GREEN)Using uv for Python dependencies$(RESET)"; \
		cd $(BACKEND_DIR) && uv pip install --system -r requirements.txt -r requirements-dev.txt; \
	else \
		echo "$(YELLOW)Using pip for Python dependencies$(RESET)"; \
		cd $(BACKEND_DIR) && pip install -r requirements.txt -r requirements-dev.txt; \
	fi
	@echo "$(GREEN)Installing frontend dependencies...$(RESET)"
	@cd $(FRONTEND_DIR) && npm ci
	@echo "$(GREEN)Installing pre-commit hooks...$(RESET)"
	@pre-commit install
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"

.PHONY: dev
dev: ## Start development environment with Docker
	@echo "$(CYAN)Starting development environment...$(RESET)"
	@if [ "$(HAS_DOCKER_COMPOSE)" ]; then \
		docker-compose -f $(DOCKER_COMPOSE_DEV) up -d; \
		echo "$(GREEN)✅ Services started!$(RESET)"; \
		echo "$(BLUE)Backend: http://localhost:8000$(RESET)"; \
		echo "$(BLUE)Frontend: http://localhost:3000$(RESET)"; \
		echo "$(BLUE)Admin: http://localhost:8000/admin/$(RESET)"; \
	else \
		echo "$(RED)❌ docker-compose not found$(RESET)"; \
		exit 1; \
	fi

.PHONY: dev-local
dev-local: ## Start development locally (without Docker)
	@echo "$(CYAN)Starting local development...$(RESET)"
	@echo "$(YELLOW)Starting backend...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py runserver 8000 &
	@echo "$(YELLOW)Starting frontend...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run dev &
	@echo "$(GREEN)✅ Local development started!$(RESET)"

.PHONY: stop
stop: ## Stop all development services
	@echo "$(CYAN)Stopping development services...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_DEV) down
	@pkill -f "python manage.py runserver" || true
	@pkill -f "npm run dev" || true
	@echo "$(GREEN)✅ All services stopped$(RESET)"

.PHONY: restart
restart: stop dev ## Restart development environment

.PHONY: logs
logs: ## Show development logs
	@docker-compose -f $(DOCKER_COMPOSE_DEV) logs -f

##@ 🧪 Testing

.PHONY: test
test: test-backend test-frontend ## Run all tests

.PHONY: test-backend
test-backend: ## Run backend tests
	@echo "$(CYAN)Running backend tests...$(RESET)"
	@cd $(BACKEND_DIR) && pytest -v --tb=short --cov=apps --cov-report=term-missing

.PHONY: test-frontend
test-frontend: ## Run frontend tests
	@echo "$(CYAN)Running frontend tests...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run test

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(RESET)"
	@cd $(BACKEND_DIR) && pytest --cov=apps --cov-report=html --cov-report=term-missing
	@cd $(FRONTEND_DIR) && npm run test:coverage
	@echo "$(GREEN)✅ Coverage reports generated$(RESET)"
	@echo "$(BLUE)Backend coverage: backend/htmlcov/index.html$(RESET)"
	@echo "$(BLUE)Frontend coverage: nuxt_frontend/coverage/index.html$(RESET)"

.PHONY: test-smoke
test-smoke: ## Run smoke tests
	@echo "$(CYAN)Running smoke tests...$(RESET)"
	@cd $(BACKEND_DIR) && python smoke_diagnostics.py
	@cd $(BACKEND_DIR) && python test_rag.py

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running E2E tests...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run test:e2e

##@ 🎨 Code Quality

.PHONY: lint
lint: lint-backend lint-frontend ## Run all linters

.PHONY: lint-backend
lint-backend: ## Lint backend code
	@echo "$(CYAN)Linting backend code...$(RESET)"
	@ruff check $(BACKEND_DIR)/ --output-format=github
	@echo "$(GREEN)✅ Backend linting completed$(RESET)"

.PHONY: lint-frontend
lint-frontend: ## Lint frontend code
	@echo "$(CYAN)Linting frontend code...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run lint
	@echo "$(GREEN)✅ Frontend linting completed$(RESET)"

.PHONY: format
format: format-backend format-frontend ## Format all code

.PHONY: format-backend
format-backend: ## Format backend code
	@echo "$(CYAN)Formatting backend code...$(RESET)"
	@ruff format $(BACKEND_DIR)/
	@ruff check $(BACKEND_DIR)/ --fix
	@echo "$(GREEN)✅ Backend formatting completed$(RESET)"

.PHONY: format-frontend
format-frontend: ## Format frontend code
	@echo "$(CYAN)Formatting frontend code...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run format
	@cd $(FRONTEND_DIR) && npm run lint:fix
	@echo "$(GREEN)✅ Frontend formatting completed$(RESET)"

.PHONY: typecheck
typecheck: typecheck-backend typecheck-frontend ## Run type checking

.PHONY: typecheck-backend
typecheck-backend: ## Run backend type checking
	@echo "$(CYAN)Running backend type checking...$(RESET)"
	@mypy $(BACKEND_DIR)/apps $(BACKEND_DIR)/core --config-file=pyproject.toml

.PHONY: typecheck-frontend
typecheck-frontend: ## Run frontend type checking
	@echo "$(CYAN)Running frontend type checking...$(RESET)"
	@cd $(FRONTEND_DIR) && npm run typecheck

.PHONY: security
security: ## Run security checks
	@echo "$(CYAN)Running security checks...$(RESET)"
	@bandit -r $(BACKEND_DIR)/ -c .bandit
	@safety check --ignore=70612
	@echo "$(GREEN)✅ Security checks completed$(RESET)"

.PHONY: check
check: lint typecheck security ## Run all code quality checks
	@echo "$(GREEN)✅ All quality checks passed!$(RESET)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit on all files
	@echo "$(CYAN)Running pre-commit hooks...$(RESET)"
	@pre-commit run --all-files

##@ 🗃️ Database

.PHONY: migrate
migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py migrate
	@echo "$(GREEN)✅ Migrations completed$(RESET)"

.PHONY: migrations
migrations: ## Create new database migrations
	@echo "$(CYAN)Creating database migrations...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py makemigrations

.PHONY: shell
shell: ## Open Django shell
	@echo "$(CYAN)Opening Django shell...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py shell_plus

.PHONY: dbshell
dbshell: ## Open database shell
	@cd $(BACKEND_DIR) && python manage.py dbshell

.PHONY: superuser
superuser: ## Create Django superuser
	@echo "$(CYAN)Creating Django superuser...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py createsuperuser

.PHONY: loaddata
loaddata: ## Load sample data
	@echo "$(CYAN)Loading sample data...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py loaddata fixtures/initial_data.json

.PHONY: dumpdata
dumpdata: ## Dump database data
	@echo "$(CYAN)Dumping database data...$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.Permission > fixtures/db_dump.json

##@ 🚢 Production

.PHONY: prod
prod: ## Start production environment
	@echo "$(CYAN)Starting production environment...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_PROD) up -d
	@echo "$(GREEN)✅ Production environment started$(RESET)"

.PHONY: prod-build
prod-build: ## Build production images
	@echo "$(CYAN)Building production images...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_PROD) build

.PHONY: prod-logs
prod-logs: ## Show production logs
	@docker-compose -f $(DOCKER_COMPOSE_PROD) logs -f

.PHONY: prod-stop
prod-stop: ## Stop production environment
	@docker-compose -f $(DOCKER_COMPOSE_PROD) down

.PHONY: backup-db
backup-db: ## Backup production database
	@echo "$(CYAN)Creating database backup...$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_PROD) exec -T db pg_dump -U postgres hydraulic_diagnostic > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created$(RESET)"

##@ 🐳 Docker

.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(CYAN)Building Docker images...$(RESET)"
	@docker build -t $(PROJECT_NAME):latest .

.PHONY: docker-clean
docker-clean: ## Clean Docker artifacts
	@echo "$(CYAN)Cleaning Docker artifacts...$(RESET)"
	@docker system prune -af --volumes
	@docker image prune -af

##@ 📝 Documentation

.PHONY: docs
docs: ## Generate documentation
	@echo "$(CYAN)Generating documentation...$(RESET)"
	@cd docs && make html

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(CYAN)Serving documentation...$(RESET)"
	@cd docs/_build/html && python -m http.server 8080

##@ 🧹 Cleanup

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf $(BACKEND_DIR)/htmlcov/
	@rm -rf $(BACKEND_DIR)/.coverage
	@rm -rf $(BACKEND_DIR)/.pytest_cache/
	@rm -rf $(FRONTEND_DIR)/coverage/
	@rm -rf $(FRONTEND_DIR)/.nuxt/
	@rm -rf $(FRONTEND_DIR)/.output/
	@rm -rf $(FRONTEND_DIR)/node_modules/.cache/
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

.PHONY: clean-all
clean-all: clean docker-clean ## Deep clean (including Docker)

##@ 📋 Information

.PHONY: status
status: ## Show project status
	@echo "$(CYAN)Project Status:$(RESET)"
	@echo "$(BLUE)=== Docker Services ===$(RESET)"
	@docker-compose -f $(DOCKER_COMPOSE_DEV) ps || echo "No services running"
	@echo "\n$(BLUE)=== Database Status ===$(RESET)"
	@cd $(BACKEND_DIR) && python manage.py showmigrations | head -20 || echo "Database not accessible"
	@echo "\n$(BLUE)=== Dependencies ===$(RESET)"
	@echo "Python: $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "Node.js: $(shell node --version 2>/dev/null || echo 'Not found')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not found')"
	@echo "uv: $(shell uv --version 2>/dev/null || echo 'Not found')"

.PHONY: urls
urls: ## Show application URLs
	@echo "$(CYAN)Application URLs:$(RESET)"
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  Frontend:  http://localhost:3000"
	@echo "  Backend:   http://localhost:8000"
	@echo "  Admin:     http://localhost:8000/admin/"
	@echo "  API Docs:  http://localhost:8000/api/docs/"
	@echo "  Flower:    http://localhost:5555 (Celery monitoring)"
	@echo "$(BLUE)Database:$(RESET)"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis:      localhost:6379"

##@ ❓ Help

.PHONY: help
help: ## Display this help
	@echo "$(CYAN)Hydraulic Diagnostic SaaS - Development Commands$(RESET)"
	@echo
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# Ensure targets don't conflict with files
.PHONY: $(shell awk '/^[a-zA-Z_-]+:/ {print $$1}' $(MAKEFILE_LIST) | sed 's/://')
