from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from celery import shared_task

from .rag_core import default_local_orchestrator

logger = logging.getLogger("apps.rag_assistant")


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=5, retry_jitter=True, max_retries=3)
def rag_build_index_task(self, documents: List[str], version: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
    """
    Celery task to build and persist local FAISS index for provided documents.
    """
    version = version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    metadata = metadata or {}

    orch = default_local_orchestrator()
    logger.info("RAG: building index", extra={"version": version, "docs": len(documents)})
    path = orch.build_and_save(documents, version=version, metadata=metadata)
    logger.info("RAG: index saved", extra={"version": version, "path": path})
    return path
