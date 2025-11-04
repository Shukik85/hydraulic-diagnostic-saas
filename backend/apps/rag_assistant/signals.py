"""Сигналы Django для приложения RAG Assistant."""

import logging

from django.core.cache import cache
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import Document, RagQueryLog, RagSystem

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Document)
def document_updated(sender, instance: Document, created: bool, **kwargs) -> None:
    """Обрабатывает обновление/создание документа.

    Args:
        sender: Класс отправителя сигнала
        instance: Экземпляр документа
        created: Флаг создания (True) или обновления (False)
        **kwargs: Дополнительные аргументы
    """
    action = "создан" if created else "обновлен"
    logger.info(f"Документ {instance.title} {action}")

    # Очистка кэша поиска конкретной системы
    cache.delete_pattern(f"rag:{instance.rag_system_id}:search:*")
    cache.delete_pattern(f"rag:{instance.rag_system_id}:faq:*")


@receiver(post_delete, sender=Document)
def document_deleted(sender, instance: Document, **kwargs) -> None:
    """Обрабатывает удаление документа.

    Args:
        sender: Класс отправителя сигнала
        instance: Экземпляр документа
        **kwargs: Дополнительные аргументы
    """
    logger.info(f"Документ {instance.title} удален")

    # Очистка кэша
    cache.delete_pattern(f"rag:{instance.rag_system_id}:search:*")
    cache.delete_pattern(f"rag:{instance.rag_system_id}:faq:*")


@receiver(post_save, sender=RagSystem)
def rag_system_updated(sender, instance: RagSystem, created: bool, **kwargs) -> None:
    """Обновляет настройки/состояние RAG системы.

    Args:
        sender: Класс отправителя сигнала
        instance: Экземпляр RAG системы
        created: Флаг создания (True) или обновления (False)
        **kwargs: Дополнительные аргументы
    """
    logger.info(f"RAG система {'создана' if created else 'обновлена'}: {instance.name}")
    cache.delete_pattern(f"rag:{instance.id}:*")


@receiver(post_delete, sender=RagSystem)
def rag_system_deleted(sender, instance: RagSystem, **kwargs) -> None:
    """Обрабатывает удаление RAG системы.

    Args:
        sender: Класс отправителя сигнала
        instance: Экземпляр RAG системы
        **kwargs: Дополнительные аргументы
    """
    logger.info(f"RAG система удалена: {instance.name}")
    cache.delete_pattern(f"rag:{instance.id}:*")


@receiver(post_save, sender=RagQueryLog)
def rag_query_logged(sender, instance: RagQueryLog, created: bool, **kwargs) -> None:
    """Создает логи запросов - можно обновлять метрики в кэше.

    Args:
        sender: Класс отправителя сигнала
        instance: Экземпляр лога запроса
        created: Флаг создания (True)
        **kwargs: Дополнительные аргументы
    """
    if created:
        cache.incr(
            f"ai_metrics:{instance.timestamp.strftime('%Y-%m-%d:%H')}:ai_requests",
            ignore_key_check=True,
        )
        logger.debug("RAG запрос залогирован")
