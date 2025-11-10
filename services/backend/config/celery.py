"""Модуль проекта с автогенерированным докстрингом."""

import os

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun
from celery.utils.log import get_task_logger
from decouple import config

# Устанавливаем Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Создаем Celery app
app = Celery("hydraulic_diagnostic")

# Конфигурация из Django settings
app.config_from_object("django.conf:settings", namespace="CELERY")

# ОПТИМИЗИРОВАННЫЕ настройки Celery
app.conf.update(
    # Основные настройки
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ
    worker_prefetch_multiplier=1,  # Предотвращает переполнение очереди
    task_acks_late=True,  # Подтверждение после выполнения
    worker_max_tasks_per_child=1000,  # Перезапуск worker каждые 1000 задач
    # Обработка ошибок и retry
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,  # 1 минута
    task_max_retries=3,
    # Маршрутизация задач по очередям
    task_routes={
        "rag_assistant.tasks.*": {"queue": "ai_tasks"},
        "diagnostics.tasks.*": {"queue": "diagnostics"},
        "users.tasks.*": {"queue": "users"},
    },
    # Мониторинг
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Оптимизация для production
    task_time_limit=config("CELERY_TASK_TIME_LIMIT", default=300, cast=int),  # 5 минут
    task_soft_time_limit=config(
        "CELERY_TASK_SOFT_TIME_LIMIT", default=240, cast=int
    ),  # 4 минуты
    worker_max_memory_per_child=config(
        "CELERY_WORKER_MAX_MEMORY_PER_CHILD", default=200000, cast=int
    ),  # 200MB
    # Heartbeat для мониторинга
    worker_heartbeat_interval=30,  # 30 секунд
    # Result backend оптимизация
    result_expires=3600,  # 1 час храним результаты
    result_persistent=True,
    # Compression
    task_compression="gzip",
    result_compression="gzip",
)

# Автообнаружение задач
app.autodiscover_tasks()

# ОПТИМИЗАЦИЯ: настройка логгирования Celery
logger = get_task_logger(__name__)


# Мониторинг состояния Celery
@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")


@task_prerun.connect
def task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds
):
    logger.info(f"Task {task.name} started: {task_id}")


@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **kwds,
):
    logger.info(f"Task {task.name} finished: {task_id} (state: {state})")


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds
):
    logger.error(f"Task {sender.name} failed: {task_id} - {exception}")
