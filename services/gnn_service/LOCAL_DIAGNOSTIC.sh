#!/bin/bash

# 🔍 LOCAL DIAGNOSTIC SCRIPT - GNN Service
# 
# Этот скрипт запускает все проверки локально и показывает результаты
# в чистом, понятном формате.
#
# Usage: bash LOCAL_DIAGNOSTIC.sh
# 
# Требует установки:
#   - pytest
#   - mypy
#   - ruff
#
# Установка:
#   pip install pytest mypy ruff pytest-asyncio pytest-cov

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔍 GNN SERVICE - LOCAL DIAGNOSTIC REPORT                 ║${NC}"
echo -e "${BLUE}║     $(date '+%Y-%m-%d %H:%M:%S')                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# SECTION 1: PYTEST - Test Suite
# ============================================================================

echo -e "${YELLOW}1️⃣  PYTEST - Test Suite${NC}"
echo -e "${YELLOW}════════════════════════════════════════════${NC}"
echo ""

if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest не установлен${NC}"
    echo "Установи: pip install pytest pytest-asyncio"
    echo ""
else
    echo -e "${GREEN}✅ pytest найден${NC}"
    echo ""
    
    echo "Running: pytest tests/ -v --tb=short --no-header"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    # Запуск тестов с capture output
    if pytest tests/ -v --tb=short --no-header 2>&1 | tee /tmp/pytest_output.txt; then
        echo -e "${GREEN}✅ Все тесты PASSED${NC}"
    else
        echo -e "${RED}❌ Некоторые тесты FAILED${NC}"
        echo ""
        echo -e "${YELLOW}FAILED TESTS:${NC}"
        grep -E "FAILED|ERROR" /tmp/pytest_output.txt || echo "No specific failures found"
    fi
    echo ""
fi

# ============================================================================
# SECTION 2: MYPY - Type Checking
# ============================================================================

echo -e "${YELLOW}2️⃣  MYPY - Type Checking (Strict Mode)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════${NC}"
echo ""

if ! command -v mypy &> /dev/null; then
    echo -e "${RED}❌ mypy не установлен${NC}"
    echo "Установи: pip install mypy"
    echo ""
else
    echo -e "${GREEN}✅ mypy найден${NC}"
    echo ""
    
    echo "Running: mypy src/ --strict --no-incremental"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    # Запуск mypy с collect errors
    if mypy src/ --strict --no-incremental 2>&1 | tee /tmp/mypy_output.txt; then
        echo -e "${GREEN}✅ Все файлы прошли mypy проверку${NC}"
    else
        echo -e "${RED}❌ Обнаружены mypy ошибки${NC}"
        echo ""
        
        # Count errors by file
        echo -e "${YELLOW}ОШИБКИ ПО ФАЙЛАМ:${NC}"
        grep "^src/" /tmp/mypy_output.txt | cut -d: -f1 | sort | uniq -c | sort -rn | while read count file; do
            echo "  $count errors: $file"
        done
        echo ""
        
        # Count errors by type
        echo -e "${YELLOW}ОШИБКИ ПО ТИПАМ:${NC}"
        grep -oE '\[(.*?)\]' /tmp/mypy_output.txt | sort | uniq -c | sort -rn | head -15 || echo "  (no specific error types found)"
        echo ""
        
        # Show first 20 errors
        echo -e "${YELLOW}ПЕРВЫЕ 20 ОШИБОК:${NC}"
        grep "^src/" /tmp/mypy_output.txt | head -20
        
        # Count total
        total_errors=$(grep -c "^src/" /tmp/mypy_output.txt || echo "0")
        echo ""
        echo -e "${RED}📊 ИТОГО: $total_errors mypy ошибок${NC}"
    fi
    echo ""
fi

# ============================================================================
# SECTION 3: RUFF - Linting & Code Quality
# ============================================================================

echo -e "${YELLOW}3️⃣  RUFF - Linting & Code Quality${NC}"
echo -e "${YELLOW}════════════════════════════════════════════${NC}"
echo ""

if ! command -v ruff &> /dev/null; then
    echo -e "${RED}❌ ruff не установлен${NC}"
    echo "Установи: pip install ruff"
    echo ""
else
    echo -e "${GREEN}✅ ruff найден${NC}"
    echo ""
    
    echo "Running: ruff check src/ tests/"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    if ruff check src/ tests/ 2>&1 | tee /tmp/ruff_output.txt; then
        echo -e "${GREEN}✅ Нет ruff ошибок${NC}"
    else
        echo -e "${YELLOW}⚠️  Обнаружены ruff нарушения${NC}"
        echo ""
        
        # Count by rule
        echo -e "${YELLOW}НАРУШЕНИЯ ПО ПРАВИЛАМ:${NC}"
        grep -oE '[A-Z][0-9]{3}' /tmp/ruff_output.txt | sort | uniq -c | sort -rn | head -10 || echo "  (no specific rules found)"
        echo ""
        
        # Show summary
        total_issues=$(wc -l < /tmp/ruff_output.txt)
        echo -e "${YELLOW}📊 ИТОГО: ~$total_issues ruff проблем${NC}"
    fi
    echo ""
fi

# ============================================================================
# SECTION 4: COVERAGE - Test Coverage Report
# ============================================================================

echo -e "${YELLOW}4️⃣  COVERAGE - Test Coverage${NC}"
echo -e "${YELLOW}════════════════════════════════════════════${NC}"
echo ""

if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest не установлен (нужен для coverage)${NC}"
else
    echo "Running: pytest tests/ --cov=src --cov-report=term-missing"
    echo -e "${BLUE}────────────────────────────────────────────${NC}"
    
    if pytest tests/ --cov=src --cov-report=term-missing --tb=line 2>&1 | tail -30; then
        echo -e "${GREEN}✅ Coverage отчет создан${NC}"
    else
        echo -e "${YELLOW}⚠️  Coverage отчет с ошибками${NC}"
    fi
fi
echo ""

# ============================================================================
# SECTION 5: SUMMARY
# ============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    📊 SUMMARY REPORT                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse results
PYTEST_PASS=0
MYPY_PASS=0
RUFF_PASS=0

if grep -q "passed" /tmp/pytest_output.txt 2>/dev/null; then
    PYTEST_PASS=1
fi

if [ -f /tmp/mypy_output.txt ] && [ ! -s /tmp/mypy_output.txt ]; then
    MYPY_PASS=1
elif grep -q "Success" /tmp/mypy_output.txt 2>/dev/null; then
    MYPY_PASS=1
fi

if [ -f /tmp/ruff_output.txt ] && [ ! -s /tmp/ruff_output.txt ]; then
    RUFF_PASS=1
elif grep -q "No issues" /tmp/ruff_output.txt 2>/dev/null; then
    RUFF_PASS=1
fi

echo -e "Status:                 Test Suite           mypy           Ruff"
echo -e "$([ $PYTEST_PASS -eq 1 ] && echo -e "${GREEN}✅ PASS${NC}" || echo -e "${RED}❌ FAIL${NC}")               $([ $MYPY_PASS -eq 1 ] && echo -e "${GREEN}✅ PASS${NC}" || echo -e "${RED}❌ FAIL${NC}")         $([ $RUFF_PASS -eq 1 ] && echo -e "${GREEN}✅ PASS${NC}" || echo -e "${RED}❌ FAIL${NC}")"
echo ""

echo -e "${YELLOW}DETAILED LOGS:${NC}"
echo "  pytest:  /tmp/pytest_output.txt"
echo "  mypy:    /tmp/mypy_output.txt"
echo "  ruff:    /tmp/ruff_output.txt"
echo ""

echo -e "${BLUE}NEXT STEPS:${NC}"
echo "  1. Review the output above carefully"
echo "  2. Understand which errors are critical vs non-critical"
echo "  3. Decide on fix strategy (Quick Fix / Critical Path / Full Cleanup)"
echo "  4. Run diagnostics again after fixes to verify"
echo ""

echo -e "${GREEN}✅ Diagnostic complete!${NC}"
echo ""
