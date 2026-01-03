#!/usr/bin/env bash
# ============================================================================
# GNN Service Test Suite Runner
# ============================================================================
# Runs pytest with coverage reporting
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run unit tests only
#   ./run_tests.sh integration  # Run integration tests only
#   ./run_tests.sh --help       # Show help
#
# Requirements:
#   - pytest
#   - pytest-cov
#   - pytest-xdist (optional, for parallel execution)
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_DIR="$SCRIPT_DIR"

# Configuration
COVERAGE_MIN=85  # Minimum coverage threshold
PYTEST_ARGS="-v --tb=short"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_CATEGORY]

Test Categories:
    all             Run all tests (default)
    unit            Run unit tests only
    integration     Run integration tests only

Options:
    -h, --help      Show this help message
    --no-cov        Skip coverage reporting
    --parallel      Run tests in parallel (requires pytest-xdist)
    --fast          Fast mode (no coverage, parallel)
    --markers       Show available pytest markers

Examples:
    $0                          # Run all tests with coverage
    $0 unit                     # Run unit tests only
    $0 --fast                   # Quick test run
    $0 --parallel integration   # Parallel integration tests

Environment Variables:
    PYTEST_ARGS     Additional pytest arguments
    COVERAGE_MIN    Minimum coverage threshold (default: 85)

EOF
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check pytest
    if ! python3 -c "import pytest" 2>/dev/null; then
        print_error "pytest not installed"
        print_info "Install: pip install pytest pytest-cov"
        exit 1
    fi
    
    # Check pytest-cov
    if [[ "$SKIP_COVERAGE" != "true" ]]; then
        if ! python3 -c "import pytest_cov" 2>/dev/null; then
            print_warning "pytest-cov not installed (coverage will be skipped)"
            print_info "Install: pip install pytest-cov"
            SKIP_COVERAGE="true"
        fi
    fi
    
    # Check pytest-xdist (optional)
    if [[ "$PARALLEL" == "true" ]]; then
        if ! python3 -c "import xdist" 2>/dev/null; then
            print_warning "pytest-xdist not installed (parallel execution disabled)"
            print_info "Install: pip install pytest-xdist"
            PARALLEL="false"
        fi
    fi
    
    print_success "All required dependencies found"
}

run_tests() {
    local test_path="$1"
    local test_name="$2"
    
    print_header "Running $test_name"
    
    cd "$SERVICE_DIR" || exit 1
    
    # Build pytest command
    local cmd="python3 -m pytest $test_path $PYTEST_ARGS"
    
    # Add coverage
    if [[ "$SKIP_COVERAGE" != "true" ]]; then
        cmd="$cmd --cov=src --cov-report=html --cov-report=term-missing"
        cmd="$cmd --cov-fail-under=$COVERAGE_MIN"
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    # Execute
    print_info "Command: $cmd"
    echo ""
    
    if eval "$cmd"; then
        print_success "$test_name completed successfully"
        return 0
    else
        print_error "$test_name failed"
        return 1
    fi
}

generate_summary() {
    print_header "Test Summary"
    
    if [[ -f "$SERVICE_DIR/htmlcov/index.html" ]]; then
        print_success "Coverage report: file://$SERVICE_DIR/htmlcov/index.html"
    fi
    
    if [[ -f "$SERVICE_DIR/.coverage" ]]; then
        print_info "Coverage data: $SERVICE_DIR/.coverage"
    fi
    
    echo ""
    print_info "Test artifacts:"
    echo "  - Coverage HTML: htmlcov/"
    echo "  - Coverage data: .coverage"
    echo "  - Pytest cache: .pytest_cache/"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse arguments
    local test_category="all"
    local SKIP_COVERAGE="false"
    local PARALLEL="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --no-cov)
                SKIP_COVERAGE="true"
                shift
                ;;
            --parallel)
                PARALLEL="true"
                shift
                ;;
            --fast)
                SKIP_COVERAGE="true"
                PARALLEL="true"
                shift
                ;;
            --markers)
                cd "$SERVICE_DIR" || exit 1
                python3 -m pytest --markers
                exit 0
                ;;
            unit|integration|all)
                test_category="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Header
    print_header "GNN Service Test Suite"
    echo ""
    print_info "Service: GNN Service"
    print_info "Directory: $SERVICE_DIR"
    print_info "Category: $test_category"
    print_info "Coverage: $([ "$SKIP_COVERAGE" = "true" ] && echo "disabled" || echo "enabled (min: ${COVERAGE_MIN}%)")"
    print_info "Parallel: $([ "$PARALLEL" = "true" ] && echo "enabled" || echo "disabled")"
    echo ""
    
    # Check dependencies
    check_dependencies
    echo ""
    
    # Run tests
    local exit_code=0
    
    case $test_category in
        unit)
            run_tests "tests/unit/" "Unit Tests" || exit_code=$?
            ;;
        integration)
            run_tests "tests/test_*integration*.py" "Integration Tests" || exit_code=$?
            ;;
        all)
            run_tests "tests/" "All Tests" || exit_code=$?
            ;;
        *)
            print_error "Unknown test category: $test_category"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    
    # Generate summary
    generate_summary
    
    # Exit
    if [[ $exit_code -eq 0 ]]; then
        echo ""
        print_success "All tests passed! 🎉"
    else
        echo ""
        print_error "Tests failed! ❌"
    fi
    
    exit $exit_code
}

main "$@"
