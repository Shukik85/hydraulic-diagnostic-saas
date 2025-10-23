"""Модуль проекта с автогенерированным докстрингом."""

import logging

from django.core.cache import cache
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import Document, RagQueryLog, RagSystem

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Document)
def document_updated(sender, instance, created, **kwargs):
    """Обработка обновления/создания документа."""
    action = "создан" if created else "обновлен"
    logger.info(f"Документ {instance.title} {action}")

    # Очистка кэша поиска конкретной системы
    cache.delete_pattern(f"rag:{instance.rag_system_id}:search:*")
    cache.delete_pattern(f"rag:{instance.rag_system_id}:faq:*")


@receiver(post_delete, sender=Document)
def document_deleted(sender, instance, **kwargs):
    """Обработка удаления документа."""
    logger.info(f"Документ {instance.title} удален")

    # Очистка кэша
    cache.delete_pattern(f"rag:{instance.rag_system_id}:search:*")
    cache.delete_pattern(f"rag:{instance.rag_system_id}:faq:*")


@receiver(post_save, sender=RagSystem)
def rag_system_updated(sender, instance, created, **kwargs):
    """Обновление настроек/состояния RAG системы."""
    logger.info(f"RAG система {'создана' if created else 'обновлена'}: {instance.name}")
    cache.delete_pattern(f"rag:{instance.id}:*")


@receiver(post_delete, sender=RagSystem)
def rag_system_deleted(sender, instance, **kwargs):
    """Удаление RAG системы."""
    logger.info(f"RAG система удалена: {instance.name}")
    cache.delete_pattern(f"rag:{instance.id}:*")


@receiver(post_save, sender=RagQueryLog)
def rag_query_logged(sender, instance, created, **kwargs):
    """Создание логов запросов — можно обновлять метрики в кэше."""
    if created:
        cache.incr(
            f"ai_metrics:{instance.timestamp.strftime('%Y-%m-%d:%H')}:ai_requests",
            ignore_key_check=True,
        )
        logger.debug("RAG запрос залогирован")
