#!/bin/bash
# Скрипт очистки старых Docker файлов и контейнеров
# Использование: bash cleanup_docker.sh

set -e

echo "🧹 Начинаем очистку Docker конфигураций..."

# 1. Останавливаем и удаляем все контейнеры
echo "📦 Останавливаем контейнеры..."
docker-compose down -v 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# 2. Удаляем старые docker-compose файлы
echo "🗑️  Удаляем старые docker-compose файлы..."
find . -name "docker-compose*.yml" -type f -delete 2>/dev/null || true
find . -name "docker-compose*.yaml" -type f -delete 2>/dev/null || true

# 3. Удаляем старые Dockerfiles (кроме нового архива)
echo "🗑️  Удаляем старые Dockerfile..."
find ./services -name "Dockerfile*" -type f -delete 2>/dev/null || true
find ./backend -name "Dockerfile*" -type f -delete 2>/dev/null || true
find ./ml_service -name "Dockerfile*" -type f -delete 2>/dev/null || true
find ./rag_service -name "Dockerfile*" -type f -delete 2>/dev/null || true
rm -f Dockerfile Dockerfile.* 2>/dev/null || true

# 4. Удаляем старые .dockerignore
echo "🗑️  Удаляем старые .dockerignore..."
find . -name ".dockerignore" -type f -delete 2>/dev/null || true

# 5. Очистка Docker кеша (опционально)
read -p "🤔 Очистить Docker build cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Очищаем Docker build cache..."
    docker builder prune -af
fi

# 6. Очистка неиспользуемых образов (опционально)
read -p "🤔 Удалить неиспользуемые Docker образы? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Удаляем неиспользуемые образы..."
    docker image prune -af
fi

# 7. Очистка volumes (ОСТОРОЖНО! Удалит данные БД)
read -p "⚠️  УДАЛИТЬ ВСЕ DOCKER VOLUMES (включая данные БД)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Удаляем Docker volumes..."
    docker volume prune -f
fi

echo "✅ Очистка завершена!"
echo ""
echo "📝 Следующие шаги:"
echo "   1. Распакуй новый docker_infrastructure.zip"
echo "   2. Запусти: docker-compose up --build"
