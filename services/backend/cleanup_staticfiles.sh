#!/bin/bash

# Django Admin Staticfiles Cleanup Script
# Удаляет устаревшие и backup файлы из staticfiles/admin/
# Version: 1.0
# Date: 2025-11-17

echo "🧹 Django Admin Staticfiles Cleanup"
echo "===================================="
echo ""

# Переходим в папку staticfiles/admin
cd "$(dirname "$0")/staticfiles/admin" || exit 1

echo "📂 Текущая директория: $(pwd)"
echo ""

# Файлы для удаления
FILES_TO_DELETE=(
    "css/custom_admin.css"
    "css/metallic_admin.css.bak"
    "js/custom_admin.js"
)

# Счетчики
deleted=0
not_found=0

echo "🗑️  Удаление устаревших файлов..."
echo ""

for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ❌ Удаляю: $file"
        rm -f "$file"
        ((deleted++))
    else
        echo "  ✅ Уже удалён: $file"
        ((not_found++))
    fi
done

echo ""
echo "✅ Готово!"
echo "   Удалено: $deleted файл(ов)"
echo "   Не найдено: $not_found файл(ов)"
echo ""

# Показываем оставшиеся CSS файлы
echo "📋 Актуальные CSS файлы:"
ls -lh css/*.css 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'

echo ""
echo "🔄 Рекомендуется выполнить:"
echo "   cd ../../"
echo "   python manage.py collectstatic --clear --noinput"
echo "   python manage.py runserver"
echo ""
echo "   В браузере: Ctrl+Shift+R (жёсткая перезагрузка)"
echo ""
