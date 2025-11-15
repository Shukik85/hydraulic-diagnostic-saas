# GNN Configuration Module

**Модуль управления версиями, паспортами и экспериментами Graph Neural Network моделей.**

## Возможности
- CRUD реестр production/архивных моделей (версия, путь, параметры, деплой)
- История запусков тренинга (config, dataset, tensorboard, статус)
- Django admin с поиском, фильтрацией, полной историей
- Подготовка к интеграции API/Orchestration

## Миграции
```bash
python manage.py makemigrations gnn_config
python manage.py migrate gnn_config
```

## Подключение
```python
INSTALLED_APPS += [
    'apps.gnn_config',
]
```

## Где смотреть?
- http://localhost:8000/admin/gnn_config/gnnmodelconfig/
- http://localhost:8000/admin/gnn_config/gnntrainingjob/

## TODO (следующие задачи)
- Интеграция с FastAPI endpoints (запуск деплоя, мониторинг обучения)
- UI-экшены для операторов
- Автоматическое обновление production-passport через webhook или polling
- Notification и логирование операций
