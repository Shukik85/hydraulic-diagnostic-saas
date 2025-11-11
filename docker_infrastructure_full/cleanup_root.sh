#!/bin/bash
# Скрипт очистки корня проекта от устаревших файлов
# Использование: bash cleanup_root.sh

set -e

echo "🧹 Очистка корня проекта..."
echo ""

# Подтверждение
read -p "⚠️  Это удалит старые Docker файлы в корне. Продолжить? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Отменено"
    exit 0
fi

echo ""
echo "🗑️  Удаляем старые Docker файлы..."

# Удаляем старые docker-compose файлы
rm -f docker-compose.yml
rm -f docker-compose.dev.yml
rm -f docker-compose.prod.yml
rm -f docker-compose.override.yml

# Удаляем старые Dockerfile
rm -f Dockerfile
rm -f Dockerfile.light

# Удаляем старый .dockerignore из корня
rm -f .dockerignore

# Удаляем старые Makefile
rm -f Makefile.docker

# Удаляем старые .env примеры (оставляем только главный .env)
rm -f .env.dev.example
rm -f .env.prod.example
# НЕ удаляем .env и .env.example (актуальные)

echo "✅ Старые Docker файлы удалены"
echo ""

echo "🗑️  Удаляем пустые/временные директории..."

# Удаляем пустые директории (если есть)
rmdir docker 2>/dev/null || true
rmdir deploy 2>/dev/null || true
rmdir certs 2>/dev/null || true
rmdir logs 2>/dev/null || true
rmdir models 2>/dev/null || true

echo "✅ Пустые директории удалены"
echo ""

echo "🗑️  Очистка устаревших конфигураций..."

# Удаляем старые конфиги (если есть)
rm -f .bandit
rm -f .editorconfig
rm -f .eslintrc.json
rm -f .prettierrc
rm -f package-lock.json

echo "✅ Устаревшие конфиги удалены"
echo ""

echo "📦 Итоговая структура корня:"
ls -la | grep -v "^\.\.$" | grep -v "^\.$"

echo ""
echo "✅ Очистка завершена!"
echo ""
echo "📝 Следующие шаги:"
echo "   1. Распакуй docker_infrastructure.zip в корень"
echo "   2. Настрой .env (если нужно)"
echo "   3. Запусти: make build && make up"
