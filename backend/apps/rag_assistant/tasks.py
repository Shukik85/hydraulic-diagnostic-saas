# apps/rag_assistant/tasks.py (typed guards for RagAssistant)
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, List

from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from celery import shared_task
from celery.utils.log import get_task_logger

from .models import Document, RagQueryLog, RagSystem
from .rag_service import RagAssistant

logger = get_task_logger(__name__)
performance_logger = logging.getLogger("performance")

MAX_BATCH_SIZE = 100
TASK_PROGRESS_UPDATE_INTERVAL = 10
MAX_RETRIES = 3
RETRY_COUNTDOWN = 60


@contextmanager
def task_performance_monitor(task_name: str, **metadata):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        performance_logger.info(
            "Celery task completed",
            extra={
                "task_name": task_name,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": time.time(),
                **metadata,
            },
        )


@shared_task(bind=True, max_retries=MAX_RETRIES)
def process_document_async(self, document_id: int, reindex: bool = False) -> Dict[str, Any]:
    _ = self.request.id
    try:
        with task_performance_monitor("process_document", document_id=document_id, reindex=reindex):
            self.update_state(state="PROGRESS", meta={"current": 0, "total": 100, "status": "Starting document processing..."})
            try:
                document = Document.objects.select_related("rag_system").get(id=document_id)
            except Document.DoesNotExist:
                logger.error(f"Document {document_id} not found")
                return {"status": "error", "error": f"Document {document_id} not found"}

            system = getattr(document, "rag_system", None)
            if not isinstance(system, RagSystem):
                logger.error("Document has no valid rag_system")
                return {"status": "error", "error": "Document has no rag_system", "document_id": document_id}

            self.update_state(state="PROGRESS", meta={"current": 20, "total": 100, "status": "Document loaded, initializing RAG assistant..."})

            assistant = RagAssistant(system)

            self.update_state(state="PROGRESS", meta={"current": 40, "total": 100, "status": "Processing and indexing document..."})

            with transaction.atomic():
                if hasattr(assistant, "index_document"):
                    assistant.index_document(document)  # type: ignore[attr-defined]

            self.update_state(state="PROGRESS", meta={"current": 80, "total": 100, "status": "Updating cache and metadata..."})

            document.updated_at = timezone.now()
            document.save(update_fields=["updated_at"])

            cache.delete_many([f"rag:{system.id}:search:*", f"rag:{system.id}:faq:*"])

            self.update_state(state="SUCCESS", meta={"current": 100, "total": 100, "status": "Document processing completed successfully"})
            logger.info(f"Document {document_id} processed successfully (reindex: {reindex})")
            return {"status": "success", "document_id": document_id, "title": document.title, "reindex": reindex, "timestamp": time.time()}
    except ValidationError as e:
        logger.error(f"Validation error processing document {document_id}: {str(e)}")
        return {"status": "error", "error": f"Validation error: {str(e)}", "document_id": document_id}
    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {str(exc)}\n{traceback.format_exc()}")
        if self.request.retries < MAX_RETRIES:
            logger.info(f"Retrying document processing {document_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=RETRY_COUNTDOWN * (self.request.retries + 1), exc=exc)
        return {"status": "error", "error": str(exc), "document_id": document_id, "retries_exhausted": True}


def _process_documents_for_system(system: RagSystem, docs, total_docs, processed, errors, self):
    try:
        assistant = RagAssistant(system)
        for doc in docs:
            try:
                with transaction.atomic():
                    if hasattr(assistant, "index_document"):
                        assistant.index_document(doc)  # type: ignore[attr-defined]
                processed["count"] += 1
                if processed["count"] % TASK_PROGRESS_UPDATE_INTERVAL == 0:
                    progress_percent = int((processed["count"] / total_docs) * 100)
                    self.update_state(state="PROGRESS", meta={"current": processed["count"], "total": total_docs, "percent": progress_percent, "status": f"Processed {processed['count']}/{total_docs} documents"})
            except Exception as e:
                error_info = {"document_id": doc.id, "document_title": doc.title, "error": str(e), "system_id": system.id}
                errors.append(error_info)
                logger.error(f"Error processing document {doc.id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error initializing RAG assistant for system {getattr(system, 'id', None)}: {str(e)}")
        for doc in docs:
            errors.append({"document_id": doc.id, "document_title": doc.title, "error": f"System error: {str(e)}", "system_id": getattr(system, "id", None)})


def _group_documents_by_system(documents):
    docs_by_system: Dict[int, Dict[str, Any]] = {}
    for doc in documents:
        sys_obj = getattr(doc, "rag_system", None)
        sys_id = getattr(sys_obj, "id", None)
        if sys_id not in docs_by_system:
            docs_by_system[sys_id] = {"system": sys_obj, "docs": []}
        docs_by_system[sys_id]["docs"].append(doc)
    return docs_by_system
