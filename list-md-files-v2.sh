#!/bin/bash

# ============================================================================
# List Markdown Files Script
# Hydraulic Diagnostic SaaS Project
# ============================================================================
# 
# Этот скрипт выводит список всех .md файлов проекта с анализом и фильтрацией
# 
# Дата: 20 ноября 2025
# Версия: 1.1.0
#
# ============================================================================

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Директории для исключения
# ============================================================================

EXCLUDE_DIRS=(
    "node_modules"
    ".venv"
    "venv"
    "env"
    ".env"
    "vendor"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
    "dist"
    "build"
    ".nuxt"
    ".output"
    ".next"
    "target"
    "bin"
    "obj"
    ".git"
)

# Построить find exclude pattern
FIND_EXCLUDE=""
for dir in "${EXCLUDE_DIRS[@]}"; do
    FIND_EXCLUDE="$FIND_EXCLUDE -path '*/$dir/*' -prune -o"
done

# ============================================================================
# Функции
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_section() {
    echo -e "${CYAN}▼ $1${NC}"
}

print_file() {
    local file=$1
    local status=$2
    local details=$3
    
    case $status in
        "✓")
            echo -e "${GREEN}$status ${CYAN}$file${NC} $details"
            ;;
        "⚠")
            echo -e "${YELLOW}$status ${CYAN}$file${NC} $details"
            ;;
        "✗")
            echo -e "${RED}$status ${CYAN}$file${NC} $details"
            ;;
        *)
            echo -e "${MAGENTA}$status ${CYAN}$file${NC} $details"
            ;;
    esac
}

get_file_size() {
    local file=$1
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo "$size"
    fi
}

get_line_count() {
    local file=$1
    if [ -f "$file" ]; then
        wc -l < "$file"
    fi
}

# ============================================================================
# ВСЕ .md файлы
# ============================================================================

print_header "ВСЕ .md файлы в проекте"

echo -e "${CYAN}Поиск всех .md файлов...${NC}"
echo -e "${YELLOW}Исключаются директории: ${EXCLUDE_DIRS[*]}${NC}"
echo ""

ALL_FILES=()
while IFS= read -r -d '' file; do
    ALL_FILES+=("$file")
done < <(eval "find . $FIND_EXCLUDE -type f -name '*.md' -print0" | sort -z)

echo -e "Найдено файлов: ${BLUE}${#ALL_FILES[@]}${NC}"
echo ""

print_section "Полный список с деталями"

for file in "${ALL_FILES[@]}"; do
    size=$(get_file_size "$file")
    lines=$(get_line_count "$file")
    echo -e "${CYAN}$file${NC}"
    echo -e "   Size: ${MAGENTA}$size${NC} | Lines: ${MAGENTA}$lines${NC}"
done

# ============================================================================
# Только ДОКУМЕНТАЦИЯ (docs/)
# ============================================================================

print_header "ДОКУМЕНТАЦИЯ (docs/)"

DOC_FILES=()
for file in "${ALL_FILES[@]}"; do
    if [[ "$file" == ./docs/* ]]; then
        DOC_FILES+=("$file")
    fi
done

if [ ${#DOC_FILES[@]} -eq 0 ]; then
    echo -e "${RED}✗ Папка docs/ не найдена или пуста${NC}"
else
    echo -e "Найдено в docs/: ${GREEN}${#DOC_FILES[@]}${NC} файлов"
    echo ""
    
    print_section "Актуальная документация ✅"
    for file in "${DOC_FILES[@]}"; do
        # Проверить что это актуальные документы
        if [[ "$file" =~ (CRITICAL_UPDATES|KNOWN_ISSUES|IMPLEMENTATION_STATUS|billing-guide-yookassa|TYPESCRIPT_BEST_PRACTICES|FRONTEND_ARCHITECTURE|TESTING_ROADMAP|MOBILE_FIRST_GUIDE|deprecated-docs)\.md ]]; then
            size=$(get_file_size "$file")
            print_file "$file" "✓" "($size)"
        fi
    done
    
    print_section "Другие документы в docs/"
    for file in "${DOC_FILES[@]}"; do
        # Проверить что это НЕ актуальные документы
        if [[ ! "$file" =~ (CRITICAL_UPDATES|KNOWN_ISSUES|IMPLEMENTATION_STATUS|billing-guide-yookassa|TYPESCRIPT_BEST_PRACTICES|FRONTEND_ARCHITECTURE|TESTING_ROADMAP|MOBILE_FIRST_GUIDE|deprecated-docs)\.md ]]; then
            size=$(get_file_size "$file")
            
            # Проверить если это deprecated или старые версии
            if [[ "$file" =~ (deprecated|OLD|BACKUP|stripe|DRAFT|WIP) ]]; then
                print_file "$file" "✗" "($size) DEPRECATED"
            else
                print_file "$file" "⚠" "($size) UNKNOWN"
            fi
        fi
    done
fi

# ============================================================================
# Backend документация (services/backend/docs/)
# ============================================================================

print_header "Backend документация (services/backend/docs/)"

BACKEND_FILES=()
for file in "${ALL_FILES[@]}"; do
    if [[ "$file" == ./services/backend/docs/* ]]; then
        BACKEND_FILES+=("$file")
    fi
done

if [ ${#BACKEND_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠ services/backend/docs/ не содержит .md файлы${NC}"
else
    echo -e "Найдено: ${BLUE}${#BACKEND_FILES[@]}${NC} файлов"
    echo ""
    
    for file in "${BACKEND_FILES[@]}"; do
        size=$(get_file_size "$file")
        
        if [[ "$file" =~ (stripe|OLD|DEPRECATED|deprecated) ]]; then
            print_file "$file" "✗" "($size) DEPRECATED"
        else
            print_file "$file" "⚠" "($size)"
        fi
    done
fi

# ============================================================================
# Frontend документация (services/frontend/docs/)
# ============================================================================

print_header "Frontend документация (services/frontend/docs/)"

FRONTEND_FILES=()
for file in "${ALL_FILES[@]}"; do
    if [[ "$file" == ./services/frontend/docs/* ]]; then
        FRONTEND_FILES+=("$file")
    fi
done

if [ ${#FRONTEND_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠ services/frontend/docs/ не содержит .md файлы${NC}"
else
    echo -e "Найдено: ${BLUE}${#FRONTEND_FILES[@]}${NC} файлов"
    echo ""
    
    for file in "${FRONTEND_FILES[@]}"; do
        size=$(get_file_size "$file")
        
        if [[ "$file" =~ (stripe|OLD|DEPRECATED|deprecated) ]]; then
            print_file "$file" "✗" "($size) DEPRECATED"
        else
            print_file "$file" "⚠" "($size)"
        fi
    done
fi

# ============================================================================
# README файлы
# ============================================================================

print_header "README файлы"

README_FILES=()
for file in "${ALL_FILES[@]}"; do
    if [[ "$file" == *"README.md" ]]; then
        README_FILES+=("$file")
    fi
done

if [ ${#README_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠ README файлы не найдены${NC}"
else
    echo -e "Найдено: ${BLUE}${#README_FILES[@]}${NC} файлов"
    echo ""
    
    for file in "${README_FILES[@]}"; do
        size=$(get_file_size "$file")
        lines=$(get_line_count "$file")
        print_file "$file" "✓" "($size, $lines lines)"
    done
fi

# ============================================================================
# Файлы с ключевыми словами
# ============================================================================

print_header "Анализ по ключевым словам"

echo -e "${CYAN}Файлы со Stripe упоминаниями:${NC}"
STRIPE_FILES=()
for file in "${ALL_FILES[@]}"; do
    if grep -q -i "stripe" "$file" 2>/dev/null; then
        STRIPE_FILES+=("$file")
        size=$(get_file_size "$file")
        print_file "$file" "✗" "($size) CONTAINS STRIPE"
    fi
done
if [ ${#STRIPE_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ Нет файлов со Stripe${NC}"
fi

echo ""
echo -e "${CYAN}Файлы с DEPRECATED упоминаниями:${NC}"
DEPRECATED_KEYWORD_FILES=()
for file in "${ALL_FILES[@]}"; do
    if grep -q -i "deprecated" "$file" 2>/dev/null; then
        DEPRECATED_KEYWORD_FILES+=("$file")
        size=$(get_file_size "$file")
        print_file "$file" "✓" "($size) MARKED DEPRECATED"
    fi
done
if [ ${#DEPRECATED_KEYWORD_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠ Нет файлов с DEPRECATED меткой${NC}"
fi

# ============================================================================
# Статистика
# ============================================================================

print_header "Статистика"

echo -e "Всего .md файлов: ${BLUE}${#ALL_FILES[@]}${NC}"
echo -e "  • В docs/: ${MAGENTA}${#DOC_FILES[@]}${NC}"
echo -e "  • В services/backend/docs/: ${MAGENTA}${#BACKEND_FILES[@]}${NC}"
echo -e "  • В services/frontend/docs/: ${MAGENTA}${#FRONTEND_FILES[@]}${NC}"
echo -e "  • README файлы: ${MAGENTA}${#README_FILES[@]}${NC}"

echo ""
echo -e "Проблемные файлы:"
echo -e "  • Со Stripe: ${RED}${#STRIPE_FILES[@]}${NC}"

# Общий размер всех файлов
TOTAL_SIZE=0
for file in "${ALL_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    fi
done
TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024 / 1024))

echo ""
echo -e "Общий размер: ${BLUE}${TOTAL_SIZE_MB} MB${NC}"

# ============================================================================
# Экспорт списка
# ============================================================================

print_header "Экспорт списка"

echo -e "${CYAN}Сохранение списков в файлы...${NC}"
echo ""

# Все файлы
echo "# All .md Files" > md_files_all.txt
echo "Generated: $(date)" >> md_files_all.txt
echo "" >> md_files_all.txt
for file in "${ALL_FILES[@]}"; do
    echo "$file" >> md_files_all.txt
done
print_file "md_files_all.txt" "✓" "Все файлы (${#ALL_FILES[@]} шт)"

# Только docs/
echo "# Documentation Files (docs/)" > md_files_docs.txt
echo "Generated: $(date)" >> md_files_docs.txt
echo "" >> md_files_docs.txt
for file in "${DOC_FILES[@]}"; do
    echo "$file" >> md_files_docs.txt
done
print_file "md_files_docs.txt" "✓" "Документация (${#DOC_FILES[@]} шт)"

# Файлы для удаления
echo "# Files to Delete" > md_files_to_delete.txt
echo "Generated: $(date)" >> md_files_to_delete.txt
echo "" >> md_files_to_delete.txt
for file in "${ALL_FILES[@]}"; do
    if [[ "$file" =~ (deprecated|OLD|BACKUP|stripe|DRAFT|WIP) ]]; then
        echo "$file" >> md_files_to_delete.txt
    fi
done
TO_DELETE=$(wc -l < md_files_to_delete.txt)
print_file "md_files_to_delete.txt" "✓" "Для удаления"

echo ""
echo -e "${GREEN}✓ Готово!${NC}"
echo ""

# ============================================================================
# Быстрые команды
# ============================================================================

print_header "Полезные команды"

echo -e "${CYAN}Просмотр списка:${NC}"
echo -e "${MAGENTA}cat md_files_all.txt${NC}"
echo ""

echo -e "${CYAN}Поиск файла:${NC}"
echo -e "${MAGENTA}grep 'filename' md_files_all.txt${NC}"
echo ""

echo -e "${CYAN}Вывести только имена (без пути):${NC}"
echo -e "${MAGENTA}grep -o '[^/]*\.md$' md_files_all.txt | sort -u${NC}"
echo ""

echo -e "${CYAN}Вывести список для bash/zsh:${NC}"
echo -e "${MAGENTA}while read f; do echo \"\$f\"; done < md_files_all.txt${NC}"
echo ""

echo -e "${CYAN}Удалить файлы из списка (ОПАСНО!):${NC}"
echo -e "${MAGENTA}while read f; do rm \"\$f\"; done < md_files_to_delete.txt${NC}"
echo ""

exit 0