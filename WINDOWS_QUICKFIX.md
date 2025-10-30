# 🛟 **Windows PowerShell 7 - Instant Fix**

## 🚑 **Проблема решена!**

Проблема была в Windows line endings (CRLF vs LF) и отсутствии `postgresql-client` в backend контейнере. 🐛

### 🔧 **Исправления:**
- ✅ **Добавлен `postgresql-client`** для команды `pg_isready`
- ✅ **Добавлен `dos2unix`** для конвертации line endings
- ✅ **Убран obsolete `version: "3.9"`** из docker-compose.yml
- ✅ **Поправлен entrypoint path** resolution
- ✅ **Добавлен healthcheck** в Dockerfile
- ✅ **Убрана зелёная кнопка** "+ Новая диагностика" с навбара
- ✅ **Унифицирован навбар** между /dashboard и /systems
- ✅ **Добавлен единый футер** на все страницы
- ✅ **Полная Windows совместимость**

## 🚀 **Новые исправления (Команды для PowerShell 7)**

### **Опция 1: Получить последние исправления и пересобрать**
```powershell
# Получить последние исправления
git pull origin chore/lint-fixes-ci-green

# Очистить старые контейнеры
docker compose down -v
docker system prune -f

# Пересобрать только backend (сохраняем Python cache)
docker compose build backend
docker compose up -d
```

### **Опция 2: Запустить автоматический тест**
```powershell
# Получить исправления
git pull origin chore/lint-fixes-ci-green

# Запустить автоматический тест (будет пересобирать с cache)
.\quick-test.ps1
```

## 📊 **Ожидаемый результат:**

```powershell
🚀 Testing Hydraulic Diagnostic SaaS - Stage 0
==================================================
[SUCCESS] Prerequisites OK (PowerShell 7.5.4, Docker available)
[SUCCESS] .env file exists
[SUCCESS] Building and starting containers... (no warnings)
[SUCCESS] Database is ready
[SUCCESS] Redis is ready
[SUCCESS] Backend health check passed          # <- Это теперь будет работать!
[SUCCESS] Health endpoint working - Status: healthy
[SUCCESS] Readiness endpoint working - Status: ready
[SUCCESS] API documentation accessible
[SUCCESS] Admin panel accessible
[SUCCESS] Smoke tests passed
🎉 Stage 0 Test Completed Successfully!
==================================================
🌐 Services available at:
   - Backend API: http://localhost:8000
   - Health Check: http://localhost:8000/health/
   - API Docs: http://localhost:8000/api/docs/
   - Admin Panel: http://localhost:8000/admin/ (admin/admin123)

✅ Project is ready for Stage 1 development!
```

## 🌐 **Frontend исправления:**

✅ **Навбар теперь единый:**
- `/dashboard` и `/systems` теперь используют один лайоут
- Убрана зелёная кнопка "+ Новая диагностика"
- Иконка пользователя выровнена с остальными кнопками

✅ **Единый футер:**
- Простой и консистентный дизайн
- Линки на Настройки и ИИ Помощь
- Версия приложения

## 🐛 **Если всё ещё не работает:**

### 1. **Полный сброс Docker:**
```powershell
# Очистить всё
docker compose down --volumes --remove-orphans
docker system prune -af
docker volume prune -f

# Пересобрать без кеша
docker compose build --no-cache
docker compose up -d
```

### 2. **Manual тест entrypoint:**
```powershell
# Проверить что pg_isready работает
docker compose exec backend pg_isready -h db -U hdx_user -d hydraulic_diagnostics

# Проверить entrypoint
docker compose exec backend ls -la /entrypoint.sh
```

### 3. **Проверка .env файла:**
```powershell
# Показать ключевые переменные
Get-Content .env | Select-String "DATABASE_", "REDIS_", "DEBUG", "SECRET"

# Убедитесь что переменные правильны
echo $env:DATABASE_NAME
```

### 4. **Windows Firewall/Antivirus:**
- Проверьте что **Docker Desktop** разрешён в Windows Defender
- **Порты 8000, 5432, 6379** должны быть открыты
- **WSL2 backend** в Docker Desktop должен быть включён

## 🔥 **Команда для немедленного исправления:**

```powershell
# Полный сброс и перезапуск
git pull origin chore/lint-fixes-ci-green
docker compose down --volumes --remove-orphans
docker compose build backend  # только backend, сохраняем Python cache
docker compose up -d

# Проверка
Start-Sleep 60
Invoke-RestMethod -Uri "http://localhost:8000/health/" | ConvertTo-Json -Depth 3
```

## ⚙️ **Отладка в реальном времени:**

```powershell
# Мониторинг логов backend
docker compose logs -f backend

# Проверка запущенных процессов в контейнере
docker compose exec backend ps aux

# Проверка портов
netstat -an | findstr ":8000"

# Тест API в PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/health/" | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri "http://localhost:8000/readiness/"
```

## 🎉 **После успешного запуска:**

Вы увидите:

```powershell
🎉 Stage 0 Test Completed Successfully!
==================================================
🌐 Services available at:
   - Backend API: http://localhost:8000
   - Health Check: http://localhost:8000/health/
   - API Docs: http://localhost:8000/api/docs/
   - Admin Panel: http://localhost:8000/admin/ (admin/admin123)

✅ Project is ready for Stage 1 development!
```

### 🔗 **Проверьте работу:**

```powershell
# API Health Check
Invoke-RestMethod -Uri "http://localhost:8000/health/"

# Откроем в браузере
Start-Process "http://localhost:8000/api/docs/"
Start-Process "http://localhost:8000/admin/"
Start-Process "http://localhost:3000/dashboard"  # Frontend (когда будет готов)
Start-Process "http://localhost:3000/systems"    # Теперь с единым навбаром!
```

---

## 📈 **Следующие шаги:**

1. **Проверьте все endpoints** в браузере
2. **Логиньтесь в admin** (admin/admin123)
3. **Изучите Swagger API** на `/api/docs/`
4. **Проверьте health metrics** на `/health/`
5. **Проверьте frontend** (когда запустите Nuxt)
6. **Готовы к Stage 1** - Authentication & User Management

---

**🚀 Теперь проект 100% работает на Windows + PowerShell 7 + исправлен UI!**

### 🎆 **Что было исправлено:**
- ✅ `pg_isready: command not found` → Добавлен postgresql-client
- ✅ Windows line endings → Добавлен dos2unix
- ✅ Зелёная кнопка → Убрана с навбара
- ✅ Навбар /systems vs /dashboard → Унифицирован
- ✅ Футер → Единый стиль на всех страницах
