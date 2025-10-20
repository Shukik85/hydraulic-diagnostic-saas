TEST_RUNNING_INSTRUCTIONS.md
text
# Инструкции по запуску тестов

1. Скопируйте `.env.example` → `.env` и заполните все переменные.
2. Установите зависимости:
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
3. Выполните миграции:
```
python manage.py migrate
```
4. Запустите тесты:
```
pytest
```
5. Для отчёта покрытия (CI):
```
pytest --cov=apps --cov-report=xml
```
6. Генерация HTML-отчёта:
```
pytest --cov=apps --cov-report=html
```
