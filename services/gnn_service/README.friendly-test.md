# Инструкция: изолированное friendly тестирование GNN моделей (GNN + RAG)

## 1. Новый endpoint в GNN-сервисе
### `/admin/model/test_inference`
- **Метод:** POST
- **Параметры:**
  - `equipment_id` (str)
  - `time_window` (dict)
  - `model_path` (str) — путь к тестовой onnx-модели (НЕ prod symlink!)
  - `sensor_data` (dict, optional)

**Пример запроса:**
```json
POST /admin/model/test_inference
{
    "equipment_id": "exc_001",
    "time_window": {"start_time": "2025-11-01T00:00:00Z", "end_time": "2025-11-13T00:00:00Z"},
    "model_path": "/app/models/test/experiment-20251115.onnx"
}
```

**Ответ:** как обычный inference, но c `from_prod: false` и model_version=test_...

---

## 2. Использование: Django/Orchestration
- Расширить GNNAdminClient метод test_inference(...)
- Запрос выполняется без риска смены prod-model
- Можно запускать батч-тесты для разных версий моделей
- Код для AB-сравнения: получить RAG reasoning для каждого gnn_result по разным моделям

**Пример для Django:**
```python
# test_result = client.test_inference(equipment_id, time_window, "/app/models/test/experiment.onnx")
```

---

## 3. Friendly-тестирование с RAG
- Получить inference output с test_inference
- Вызвать RAG `/interpret/diagnosis`, передать test-инференс
- Анализировать и сравнивать ответы reasoning для prod и тестовой версии

**AB-сценарий:**
```python
# 1. Получаем prod_result и test_result
prod = client.get_inference(...)
test = client.test_inference(...)
# 2. Отправляем оба в RAG
prod_diag = rag_client.interpret_diagnosis(gnn_result=prod, ...)
test_diag = rag_client.interpret_diagnosis(gnn_result=test, ...)
# 3. Сравниваем рекомендацию, reasoning, prognosis
```

---

## 4. Безопасность и best practices
- Нельзя передавать путь до prod-ссылки (будет ошибка)
- Желательно ограничить worker-ресурсы для тест-инференса (disk, gpu)
- Все логи будут отображать откуда пришёл запрос (request_id)
- Можно запускать циклические тесты для CI/CD и R&D

---

**Рекомендуется добавить тесты (pytest, scripts/integration_tests/) для автоматизации friendly проверки при каждом обновлении модели.**
