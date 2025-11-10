"""Celery задачи для построения индексов RAG системы."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

from celery import shared_task

from .rag_core import default_local_orchestrator

logger = logging.getLogger("apps.rag_assistant")


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_jitter=True,
    max_retries=3,
)
def rag_build_index_task(
    self: Any,
    documents: list[str],
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Celery задача для построения и сохранения локального FAISS индекса.

    Args:
        self: Контекст задачи Celery
        documents: Список документов для индексации
        version: Версия индекса (опционально)
        metadata: Метаданные индекса (опционально)

    Returns:
        Путь к сохраненному индексу
    """
    version = version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    metadata = metadata or {}

    orch = default_local_orchestrator()
    logger.info(
        "RAG: building index", extra={"version": version, "docs": len(documents)}
    )
    path = orch.build_and_save(documents, version=version, metadata=metadata)
    logger.info("RAG: index saved", extra={"version": version, "path": path})
    return path
