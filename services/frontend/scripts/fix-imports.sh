#!/bin/bash

# fix-imports.sh
# Автоматическое исправление импортов в Vue файлах для Nuxt 4

set -euo pipefail

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[0;31m"
COLOR_BLUE="\033[0;34m"

log_info() {
  echo -e "${COLOR_BLUE}ℹ️  $1${COLOR_RESET}"
}

log_success() {
  echo -e "${COLOR_GREEN}✅ $1${COLOR_RESET}"
}

log_warning() {
  echo -e "${COLOR_YELLOW}⚠️  $1${COLOR_RESET}"
}

log_error() {
  echo -e "${COLOR_RED}❌ $1${COLOR_RESET}"
}

log_info "🚀 Запуск автоматического исправления импортов в Vue файлах..."

cd "$(dirname "$0")/.."

COUNT=0
FIXED=0

log_info "🔍 Поиск Vue файлов с <script setup lang='ts'>..."

# Найти все .vue файлы
while IFS= read -r file; do
  COUNT=$((COUNT + 1))

  # Проверить, есть ли <script setup lang="ts">
  if ! grep -q '<script setup lang="ts">' "$file"; then
    continue
  fi

  log_info "🔧 Обработка: $file"

  # Создать временный файл
  TEMP_FILE="${file}.tmp"
  NEEDS_FIX=false
  HAS_IMPORTS=false
  IMPORT_LINE=""

  # Список Vue API для импорта
  declare -A VUE_APIS
  VUE_APIS["ref"]="used"
  VUE_APIS["computed"]="used"
  VUE_APIS["watch"]="used"
  VUE_APIS["watchEffect"]="used"
  VUE_APIS["onMounted"]="used"
  VUE_APIS["onUnmounted"]="used"
  VUE_APIS["onBeforeMount"]="used"
  VUE_APIS["onBeforeUnmount"]="used"
  VUE_APIS["onUpdated"]="used"
  VUE_APIS["onErrorCaptured"]="used"
  VUE_APIS["nextTick"]="used"
  VUE_APIS["defineProps"]="used"
  VUE_APIS["defineEmits"]="used"
  VUE_APIS["defineExpose"]="used"
  VUE_APIS["toRef"]="used"
  VUE_APIS["toRefs"]="used"
  VUE_APIS["reactive"]="used"

  # Найти, какие API используются в файле
  USED_APIS=()
  for api in "${!VUE_APIS[@]}"; do
    if grep -qE "\b${api}\(" "$file" || grep -qE "\b${api} " "$file"; then
      USED_APIS+=("$api")
    fi
  done

  # Проверить, есть ли уже импорты
  if grep -q "import.*from '#imports'" "$file" || grep -q "import.*from ['\"]vue['\"]" "$file"; then
    HAS_IMPORTS=true
  fi

  # Если используются API и нет импортов
  if [ ${#USED_APIS[@]} -gt 0 ] && [ "$HAS_IMPORTS" = false ]; then
    NEEDS_FIX=true
    IMPORT_LINE="import { $(IFS=, ; echo "${USED_APIS[*]}") } from 'vue'"
  fi

  if [ "$NEEDS_FIX" = true ]; then
    # Добавить импорт после <script setup lang="ts">
    awk -v import_line="$IMPORT_LINE" '
      /<script setup lang="ts">/ {
        print
        print import_line
        print ""
        next
      }
      { print }
    ' "$file" > "$TEMP_FILE"

    mv "$TEMP_FILE" "$file"
    log_success "✅ Исправлен: $file (добавлено: ${USED_APIS[*]})"
    FIXED=$((FIXED + 1))
  fi

done < <(find . -name '*.vue' -type f | grep -v 'node_modules' | grep -v '.nuxt')

log_info ""
log_info "📊 Статистика:"
log_info "  Обработано файлов: $COUNT"
log_success "  Исправлено: $FIXED"

if [ $FIXED -gt 0 ]; then
  log_success "🎉 Импорты успешно исправлены!"
  log_warning "⚠️  Пожалуйста, проверьте изменения вручную"
  log_info "🔍 Запустите 'npm run typecheck' для проверки"
else
  log_success "🎉 Все файлы уже корректны!"
fi
