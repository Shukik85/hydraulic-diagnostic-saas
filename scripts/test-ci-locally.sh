#!/bin/bash
# Local CI/CD simulation script
# Runs all checks that would run in GitHub Actions

set -e  # Exit on error

echo "======================================"
echo "🧪 Running Local CI/CD Checks"
echo "======================================"

BACKEND_DIR="services/backend"
FRONTEND_DIR="services/frontend"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if in project root
if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

# Stage 1: Code Quality
echo -e "\n${YELLOW}Stage 1: Code Quality${NC}"
echo "--------------------------------------"

cd "$BACKEND_DIR"

echo "Running Ruff linter..."
ruff check . || { echo -e "${RED}✗ Linting failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Linting passed${NC}"

echo "Checking code formatting..."
ruff format --check . || { echo -e "${RED}✗ Formatting check failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Formatting check passed${NC}"

echo "Running MyPy type checker..."
mypy apps/ config/ --config-file=pyproject.toml || { echo -e "${RED}✗ Type checking failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Type checking passed${NC}"

cd ../..

# Stage 2: Security
echo -e "\n${YELLOW}Stage 2: Security Checks${NC}"
echo "--------------------------------------"

cd "$BACKEND_DIR"

echo "Running Bandit security scanner..."
bandit -r apps/ config/ -ll || { echo -e "${YELLOW}! Security warnings found${NC}"; }

echo "Checking dependencies for vulnerabilities..."
safety check || { echo -e "${YELLOW}! Vulnerable dependencies found${NC}"; }

cd ../..

# Stage 3: Backend Tests
echo -e "\n${YELLOW}Stage 3: Backend Tests${NC}"
echo "--------------------------------------"

cd "$BACKEND_DIR"

export DATABASE_HOST="localhost"
export DATABASE_NAME="test_db"
export DATABASE_USER="postgres"
export DATABASE_PASSWORD="postgres"
export REDIS_URL="redis://localhost:6379/0"
export DJANGO_SECRET_KEY="test-key"
export DEBUG="False"

echo "Running tests with coverage..."
pytest --cov=apps --cov-report=term-missing --cov-fail-under=80 || {
    echo -e "${RED}✗ Tests failed or coverage below 80%${NC}"
    exit 1
}

echo -e "${GREEN}✓ All tests passed${NC}"

cd ../..

# Stage 4: Frontend Tests (if exists)
if [ -d "$FRONTEND_DIR" ]; then
    echo -e "\n${YELLOW}Stage 4: Frontend Tests${NC}"
    echo "--------------------------------------"

    cd "$FRONTEND_DIR"

    if [ -f "package.json" ]; then
        echo "Running ESLint..."
        npm run lint || { echo -e "${YELLOW}! Linting warnings${NC}"; }

        echo "Running type check..."
        npm run type-check || { echo -e "${RED}✗ Type check failed${NC}"; exit 1; }

        echo "Running unit tests..."
        npm run test:unit || { echo -e "${RED}✗ Frontend tests failed${NC}"; exit 1; }

        echo -e "${GREEN}✓ Frontend checks passed${NC}"
    fi

    cd ../..
fi

# Summary
echo -e "\n======================================"
echo -e "${GREEN}✓ All CI/CD checks passed!${NC}"
echo -e "======================================"
echo -e "\nYou can safely push your changes."
