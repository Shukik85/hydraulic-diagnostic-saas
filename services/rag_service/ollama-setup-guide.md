# 🚀 RAG Service Quick Start с Ollama

## Шаг 1: Установка Ollama

### Windows:
```bash
# Скачай и установи Ollama
# https://ollama.ai/download/windows

# Или через winget
winget install Ollama.Ollama
```

### Проверка установки:
```bash
ollama --version
```

---

## Шаг 2: Запуск Ollama

```bash
# Запустить Ollama сервер (автоматически при установке)
ollama serve

# Проверить, что работает
curl http://localhost:11434/api/tags
```

---

## Шаг 3: Скачать модель

```bash
# Легкая модель для CPU (1.5GB)
ollama pull deepseek-r1:1.5b

# Или более мощная для GPU (7GB)
ollama pull deepseek-r1:7b

# Проверить установленные модели
ollama list
```

---

## Шаг 4: Установка Python зависимостей

```bash
cd services/rag_service

# Скопировать новый requirements.txt
cp rag-requirements.txt requirements.txt

# Установить зависимости
pip install -r requirements.txt
```

---

## Шаг 5: Обновить код

### Замени `model_loader.py`:
```bash
# Скопировать новый файл
cp model_loader_ollama.py model_loader.py
```

### Обнови `config.py`:
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Ollama settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek-r1:1.5b"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 2048
```

---

## Шаг 6: Запуск RAG Service

```bash
cd services/rag_service

# Запустить сервис
python main.py
```

Откроется на `http://localhost:8004`

---

## Шаг 7: Тестирование

```bash
# Health check
curl http://localhost:8004/health

# Test generation
curl -X POST http://localhost:8004/api/v1/interpret \
  -H "Content-Type: application/json" \
  -d '{
    "gnn_output": {
      "anomaly_detected": true,
      "confidence": 0.92,
      "component_id": "valve-1"
    },
    "query": "Что случилось с клапаном?"
  }'
```

---

## 🎯 Модели для разных задач:

### CPU-friendly (легкие):
```bash
ollama pull deepseek-r1:1.5b      # 1.5GB - самая легкая
ollama pull llama3.2:3b            # 3GB
ollama pull phi4:3.8b              # 3.8GB
```

### GPU (мощные):
```bash
ollama pull deepseek-r1:7b         # 7GB
ollama pull llama3.3:70b           # 70GB (требует много VRAM)
```

---

## 🐛 Troubleshooting:

### Ollama не запускается:
```bash
# Убедись, что порт 11434 свободен
netstat -ano | findstr :11434

# Перезапусти Ollama
taskkill /F /IM ollama.exe
ollama serve
```

### Модель не загружается:
```bash
# Проверь место на диске
dir C:\Users\%USERNAME%\.ollama\models

# Очисти старые модели
ollama rm old-model-name
```

### Ошибка импорта ollama:
```bash
pip install --upgrade ollama httpx
```

---

## ✅ Готово!

RAG Service теперь работает с Ollama:
- ✅ Локальный запуск без GPU
- ✅ Быстрая генерация
- ✅ Полная конфиденциальность
- ✅ Production-ready

**Следующий шаг:** Запусти Frontend и проверь полный flow! 🚀
