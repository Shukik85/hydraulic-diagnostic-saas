#!/bin/bash

# 🔍 LOCAL DIAGNOSTIC SCRIPT - GNN Service (PRODUCTION-GRADE)
# 
# Production-ready diagnostic with proper error handling, temp file management,
# and reliable status detection.
#
# Features:
#   ✅ Secure temporary file handling (mktemp)
#   ✅ Proper exit code detection
#   ✅ No redundant pytest runs
#   ✅ Comprehensive error handling
#   ✅ TOCTOU-safe operations
#   ✅ Automatic cleanup
#
# Usage: bash LOCAL_DIAGNOSTIC.sh
# 
# Requirements:
#   pip install pytest pytest-asyncio pytest-cov mypy ruff

set -euo pipefail  # Exit on error, unset vars, pipe failures

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Status variables
PYTEST_EXIT=0
MYPY_EXIT=0
RUFF_EXIT=0

# Create secure temporary files
PYTEST_LOG=$(mktemp)
MYPY_LOG=$(mktemp)
RUFF_LOG=$(mktemp)
COVERAGE_LOG=$(mktemp)

# Cleanup on exit (trap must be set EARLY)
cleanup() {
    local exit_code=$?
    rm -f "$PYTEST_LOG" "$MYPY_LOG" "$RUFF_LOG" "$COVERAGE_LOG"
    return $exit_code
}
trap cleanup EXIT

# ============================================================================
# PREFLIGHT CHECKS
# ============================================================================

if [ ! -d "src" ]; then
    echo -e "${RED}❌ Директория src/ не найдена${NC}"
    echo "Пожалуйста, запустите скрипт из директории gnn_service"
    exit 1
fi

if [ ! -d "tests" ]; then
    echo -e "${RED}❌ Директория tests/ не найдена${NC}"
    echo "Пожалуйста, запустите скрипт из директории gnn_service"
    exit 1
fi

# ============================================================================
# HEADER
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔍 GNN SERVICE - LOCAL DIAGNOSTIC REPORT                     ║${NC}"
echo -e "${BLUE}║     $(date '+%Y-%m-%d %H:%M:%S')                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# SECTION 1: PYTEST - Test Suite (WITH COVERAGE)
# ============================================================================

echo -e "${YELLOW}1️⃣  PYTEST - Test Suite (including coverage)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════${NC}"
echo ""

if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest не установлен${NC}"
    echo "Установите: pip install pytest pytest-asyncio pytest-cov"
    echo ""
    PYTEST_EXIT=1
else
    echo -e "${GREEN}✅ pytest найден${NC}"
    echo ""
    
    echo "Running: pytest tests/ -v --tb=short --cov=src --cov-report=term-missing"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────${NC}"
    
    # Run pytest ONCE with coverage - capture exit code
    if pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --no-header > "$PYTEST_LOG" 2>&1; then
        PYTEST_EXIT=0
        echo -e "${GREEN}✅ Все тесты PASSED${NC}"
    else
        PYTEST_EXIT=$?
        echo -e "${RED}❌ Некоторые тесты FAILED${NC}"
    fi
    
    echo ""
    
    # Show test summary
    echo -e "${YELLOW}ТЕСТ-СВОДКА:${NC}"
    if grep -E "^(PASSED|FAILED|ERROR)" "$PYTEST_LOG" | head -20; then
        echo ""
    fi
    
    # Show coverage if present
    if grep -q "coverage" "$PYTEST_LOG"; then
        echo -e "${YELLOW}COVERAGE ОТЧЁТ:${NC}"
        # Extract and show only the coverage summary section
        sed -n '/^Name /,/^TOTAL/p' "$PYTEST_LOG" | tail -20
    fi
    
    echo ""
fi

# ============================================================================
# SECTION 2: MYPY - Type Checking
# ============================================================================

echo -e "${YELLOW}2️⃣  MYPY - Type Checking (Strict Mode)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════${NC}"
echo ""

if ! command -v mypy &> /dev/null; then
    echo -e "${RED}❌ mypy не установлен${NC}"
    echo "Установите: pip install mypy"
    echo ""
    MYPY_EXIT=1
else
    echo -e "${GREEN}✅ mypy найден${NC}"
    echo ""
    
    echo "Running: mypy src/ --strict --no-incremental"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────${NC}"
    
    # Run mypy - capture exit code (mypy returns 0 on success, 1 on errors)
    if mypy src/ --strict --no-incremental > "$MYPY_LOG" 2>&1; then
        MYPY_EXIT=0
        echo -e "${GREEN}✅ Все файлы прошли mypy проверку${NC}"
    else
        MYPY_EXIT=$?
        echo -e "${RED}❌ Обнаружены mypy ошибки${NC}"
    fi
    
    echo ""
    
    # Parse and show statistics
    if [ $MYPY_EXIT -ne 0 ]; then
        echo -e "${YELLOW}ОШИБКИ ПО ФАЙЛАМ:${NC}"
        grep "^src/" "$MYPY_LOG" | cut -d: -f1 | sort | uniq -c | sort -rn | while read count file; do
            echo "  $count ошибок: $file"
        done
        echo ""
        
        echo -e "${YELLOW}ОШИБКИ ПО ТИПАМ:${NC}"
        grep -oE '\[[a-z-]+\]' "$MYPY_LOG" | sort | uniq -c | sort -rn | head -15 | while read count type; do
            echo "  $count: $type"
        done
        echo ""
        
        # Show first 20 errors for investigation
        echo -e "${YELLOW}ПЕРВЫЕ 20 ОШИБОК:${NC}"
        grep "^src/" "$MYPY_LOG" | head -20 | while read line; do
            echo "  $line"
        done
        
        # Count total
        total_errors=$(grep -c "^src/" "$MYPY_LOG" || echo "0")
        echo ""
        echo -e "${RED}📊 ИТОГО: $total_errors mypy ошибок${NC}"
    fi
    
    echo ""
fi

# ============================================================================
# SECTION 3: RUFF - Linting & Code Quality
# ============================================================================

echo -e "${YELLOW}3️⃣  RUFF - Linting & Code Quality${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════════${NC}"
echo ""

if ! command -v ruff &> /dev/null; then
    echo -e "${RED}❌ ruff не установлен${NC}"
    echo "Установите: pip install ruff"
    echo ""
    RUFF_EXIT=1
else
    echo -e "${GREEN}✅ ruff найден${NC}"
    echo ""
    
    echo "Running: ruff check src/ tests/"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────${NC}"
    
    # Run ruff - capture exit code (ruff returns 0 on success, 1 on errors)
    if ruff check src/ tests/ > "$RUFF_LOG" 2>&1; then
        RUFF_EXIT=0
        echo -e "${GREEN}✅ Нет ruff ошибок${NC}"
    else
        RUFF_EXIT=$?
        echo -e "${YELLOW}⚠️  Обнаружены ruff нарушения${NC}"
    fi
    
    echo ""
    
    if [ $RUFF_EXIT -ne 0 ] && [ -s "$RUFF_LOG" ]; then
        echo -e "${YELLOW}НАРУШЕНИЯ ПО ПРАВИЛАМ:${NC}"
        grep -oE '[A-Z][0-9]{3}' "$RUFF_LOG" | sort | uniq -c | sort -rn | head -10 | while read count rule; do
            echo "  $count: $rule"
        done
        echo ""
        
        # Show total issues
        total_issues=$(wc -l < "$RUFF_LOG")
        echo -e "${YELLOW}📊 ИТОГО: ~$total_issues ruff проблем${NC}"
    fi
    
    echo ""
fi

# ============================================================================
# SECTION 4: SUMMARY
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    📋 SUMMARY REPORT                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Determine status based on EXIT CODES (not grepping for strings)
echo -e "Status:\n"

echo -n "Test Suite:  "
if [ $PYTEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL (exit code: $PYTEST_EXIT)${NC}"
fi

echo -n "Type Check:  "
if [ $MYPY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL (exit code: $MYPY_EXIT)${NC}"
fi

echo -n "Linting:     "
if [ $RUFF_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  ISSUES FOUND (exit code: $RUFF_EXIT)${NC}"
fi

echo ""
echo -e "${BLUE}DETAILED LOGS:${NC}"
echo "  📄 pytest: $PYTEST_LOG"
echo "  📄 mypy:   $MYPY_LOG"
echo "  📄 ruff:   $RUFF_LOG"
echo ""
echo "These files will be automatically cleaned up on exit."
echo ""

# Overall status
echo -e "${BLUE}OVERALL STATUS:${NC}"
if [ $PYTEST_EXIT -eq 0 ] && [ $MYPY_EXIT -eq 0 ] && [ $RUFF_EXIT -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED - Ready for deployment!${NC}"
    exit 0
elif [ $PYTEST_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Tests failed - review output and fix issues${NC}"
    exit 1
elif [ $MYPY_EXIT -ne 0 ]; then
    echo -e "${RED}⚠️  Type errors detected - review mypy output${NC}"
    exit 1
else
    echo -e "${YELLOW}⚠️  Linting issues found - not critical but should review${NC}"
    exit 0  # Ruff issues are non-blocking
fi
