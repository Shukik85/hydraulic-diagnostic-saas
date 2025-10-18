import logging

from django.core.cache import cache
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import DocumentChunk, KnowledgeBase, RAGSystemSettings

logger = logging.getLogger(__name__)


@receiver(post_save, sender=KnowledgeBase)
def knowledge_base_updated(sender, instance, created, **kwargs):
    """Обработка обновления документа в базе знаний"""
    action = "создан" if created else "обновлен"
    logger.info(f"Документ {instance.title} {action}")

    # Очистка кэша поиска
    cache.delete_pattern("rag_search_*")

    # Обновление статистики
    if created:
        cache.delete("knowledge_base_stats")


@receiver(post_delete, sender=KnowledgeBase)
def knowledge_base_deleted(sender, instance, **kwargs):
    """Обработка удаления документа"""
    logger.info(f"Документ {instance.title} удален")

    # Очистка кэша
    cache.delete_pattern("rag_search_*")
    cache.delete("knowledge_base_stats")


@receiver(post_save, sender=RAGSystemSettings)
def rag_settings_updated(sender, instance, created, **kwargs):
    """Обработка обновления настроек RAG системы"""
    if instance.is_active:
        # Деактивировать все остальные настройки
        RAGSystemSettings.objects.exclude(id=instance.id).update(is_active=False)

        # Очистка кэша моделей
        cache.delete("rag_embedding_model")

        logger.info(f"Настройки RAG системы {'созданы' if created else 'обновлены'}")


@receiver(post_save, sender=DocumentChunk)
def document_chunk_created(sender, instance, created, **kwargs):
    """Обработка создания нового фрагмента документа"""
    if created:
        # Обновление счетчика фрагментов у документа
        chunks_count = DocumentChunk.objects.filter(document=instance.document).count()

        # Можно добавить логику обновления статистики
        logger.debug(
            f"Создан фрагмент {instance.chunk_index} для документа {instance.document.title}"
        )
