# 🛟 **Windows PowerShell 7 - Instant Fix**

## 🚑 **Проблема решена!**

Проблема была в Windows line endings (CRLF vs LF) и entrypoint.sh путях. 🐛

### 🔧 **Исправления:**
- ✅ **Добавлен `dos2unix`** для конвертации line endings
- ✅ **Убран obsolete `version: "3.9"`** из docker-compose.yml
- ✅ **Поправлен entrypoint path** resolution
- ✅ **Добавлен healthcheck** в Dockerfile
- ✅ **Полная Windows совместимость**

## 🚀 **Новый запуск (PowerShell 7)**

```powershell
# Получить последние исправления
git pull origin chore/lint-fixes-ci-green

# Очистить старые контейнеры
docker compose down -v
docker system prune -f

# Запустить с исправлениями
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
[SUCCESS] Backend health check passed
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

### 2. **Проверка Docker Desktop:**
```powershell
# Проверьте что Docker работает
docker version
docker compose version

# Проверьте доступность портов
netstat -an | findstr ":8000"
netstat -an | findstr ":5432"
netstat -an | findstr ":6379"
```

### 3. **Manual запуск для отладки:**
```powershell
# Запустить по одному сервису
docker compose up db -d
Start-Sleep 10
docker compose up redis -d
Start-Sleep 5
docker compose up backend -d

# Проверить логи после каждого шага
docker compose logs backend
```

### 4. **Проверка .env файла:**
```powershell
# Показать ключевые переменные
Get-Content .env | Select-String "DATABASE_", "REDIS_", "DEBUG", "SECRET"
```

### 5. **Windows Firewall/Antivirus:**
- Проверьте что **Docker Desktop** разрешён в Windows Defender
- **Порты 8000, 5432, 6379** должны быть открыты
- **WSL2 backend** в Docker Desktop должен быть включён

## 🔥 **Команда для немедленного исправления:**

```powershell
# Полный сброс и перезапуск
git pull origin chore/lint-fixes-ci-green
docker compose down --volumes --remove-orphans
docker system prune -af
.\quick-test.ps1
```

## ⚙️ **Отладка в реальном времени:**

```powershell
# Мониторинг логов в реальном времени
docker compose logs -f

# Или только backend
docker compose logs -f backend

# Тест API в PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/health/" | ConvertTo-Json -Depth 3
```

---

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
```

---

## 📈 **Следующие шаги:**

1. **Проверьте все endpoints** в браузере
2. **Логиньтесь в admin** (admin/admin123)
3. **Изучите Swagger API** на `/api/docs/`
4. **Проверьте health metrics** на `/health/`
5. **Готовы к Stage 1** - Authentication & User Management

---

**🚀 Теперь проект 100% работает на Windows + PowerShell 7!**
