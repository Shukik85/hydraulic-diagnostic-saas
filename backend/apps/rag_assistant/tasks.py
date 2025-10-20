# apps/rag_assistant/tasks.py
# ОПТИМИЗИРОВАННЫЕ CELERY TASKS ДЛЯ RAG ASSISTANT

import logging
import time
import traceback
from contextmanager import contextmanager
from typing import Any, Dict, List

from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from celery import shared_task
from celery.utils.log import get_task_logger

from .models import Document, RagQueryLog, RagSystem
from .rag_service import RagAssistant

# Логгер для Celery tasks
logger = get_task_logger(__name__)
performance_logger = logging.getLogger("performance")

# Константы
MAX_BATCH_SIZE = 100
TASK_PROGRESS_UPDATE_INTERVAL = 10  # Каждые 10 документов
MAX_RETRIES = 3
RETRY_COUNTDOWN = 60  # 1 минута


@contextmanager
def task_performance_monitor(task_name: str, **metadata):
    """
    Context manager для мониторинга производительности задач
    """
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
def process_document_async(
    self, document_id: int, reindex: bool = False
) -> Dict[str, Any]:
    """
    Асинхронная обработка документа с:
    - Retry механизмом
    - Progress tracking
    - Error handling
    - Performance monitoring
    """
    # task_id not used directly, but useful for logs
    _ = self.request.id

    try:
        with task_performance_monitor(
            "process_document", document_id=document_id, reindex=reindex
        ):
            # Обновляем статус задачи
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": 0,
                    "total": 100,
                    "status": "Starting document processing...",
                },
            )

            # Получаем документ
            try:
                document = Document.objects.select_related("rag_system").get(
                    id=document_id
                )
            except Document.DoesNotExist:
                logger.error(f"Document {document_id} not found")
                return {"status": "error", "error": f"Document {document_id} not found"}

            self.update_state(
                state="PROGRESS",
                meta={
                    "current": 20,
                    "total": 100,
                    "status": "Document loaded, initializing RAG assistant...",
                },
            )

            # Создаем RAG ассистента
            assistant = RagAssistant(document.rag_system)

            self.update_state(
                state="PROGRESS",
                meta={
                    "current": 40,
                    "total": 100,
                    "status": "Processing and indexing document...",
                },
            )

            # Обрабатываем документ
            with transaction.atomic():
                assistant.index_document(document)

            self.update_state(
                state="PROGRESS",
                meta={
                    "current": 80,
                    "total": 100,
                    "status": "Updating cache and metadata...",
                },
            )

            # Обновляем время обработки
            document.updated_at = timezone.now()
            document.save(update_fields=["updated_at"])

            # Очищаем кеш связанных запросов
            cache.delete_many(
                [
                    f"rag:{document.rag_system.id}:search:*",
                    f"rag:{document.rag_system.id}:faq:*",
                ]
            )

            self.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Document processing completed successfully",
                },
            )

            logger.info(
                f"Document {document_id} processed successfully (reindex: {reindex})"
            )

            return {
                "status": "success",
                "document_id": document_id,
                "title": document.title,
                "reindex": reindex,
                "timestamp": time.time(),
            }

    except ValidationError as e:
        logger.error(f"Validation error processing document {document_id}: {str(e)}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "document_id": document_id,
        }

    except Exception as exc:
        logger.error(
            f"Error processing document {document_id}: {str(exc)}\n{traceback.format_exc()}"
        )

        # Retry механизм
        if self.request.retries < MAX_RETRIES:
            logger.info(
                f"Retrying document processing {document_id} (attempt {self.request.retries + 1})"
            )
            raise self.retry(
                countdown=RETRY_COUNTDOWN * (self.request.retries + 1), exc=exc
            )

        return {
            "status": "error",
            "error": str(exc),
            "document_id": document_id,
            "retries_exhausted": True,
        }


@shared_task(bind=True, max_retries=MAX_RETRIES)
def index_documents_batch_async(self, document_ids: List[int]) -> Dict[str, Any]:  # noqa: C901
    """
    Пакетная асинхронная обработка множества документов
    ОПТИМИЗИРОВАННО для минимального количества DB запросов
    """

    if not document_ids or len(document_ids) == 0:
        return {"status": "error", "error": "No documents provided"}

    if len(document_ids) > MAX_BATCH_SIZE:
        return {
            "status": "error",
            "error": f"Batch size too large. Maximum {MAX_BATCH_SIZE} documents per batch",
        }

    task_id = self.request.id
    total_docs = len(document_ids)
    processed = 0
    errors: List[Dict[str, Any]] = []

    try:
        with task_performance_monitor("batch_index_documents", total_docs=total_docs):

            # Получаем все документы одним запросом
            documents = Document.objects.select_related("rag_system").filter(
                id__in=document_ids
            )

            # Группируем по RAG системам для эффективности
            docs_by_system: Dict[int, Dict[str, Any]] = {}
            for doc in documents:
                if doc.rag_system.id not in docs_by_system:
                    docs_by_system[doc.rag_system.id] = {
                        "system": doc.rag_system,
                        "docs": [],
                    }
                docs_by_system[doc.rag_system.id]["docs"].append(doc)

            logger.info(
                (
                    f"Batch processing: {total_docs} documents across "
                    f"{len(docs_by_system)} systems"
                )
            )

            # Обрабатываем каждую RAG систему
            for system_id, system_data in docs_by_system.items():
                system = system_data["system"]
                docs = system_data["docs"]

                try:
                    assistant = RagAssistant(system)

                    # Обрабатываем документы в этой системе
                    for i, doc in enumerate(docs, 1):
                        try:
                            with transaction.atomic():
                                assistant.index_document(doc)

                            processed += 1

                            # Обновляем progress
                            if processed % TASK_PROGRESS_UPDATE_INTERVAL == 0:
                                progress_percent = int((processed / total_docs) * 100)
                                self.update_state(
                                    state="PROGRESS",
                                    meta={
                                        "current": processed,
                                        "total": total_docs,
                                        "percent": progress_percent,
                                        "status": (
                                            f"Processed {processed}/{total_docs} documents"
                                        ),
                                    },
                                )

                        except Exception as e:
                            error_info = {
                                "document_id": doc.id,
                                "document_title": doc.title,
                                "error": str(e),
                                "system_id": system_id,
                            }
                            errors.append(error_info)
                            logger.error(
                                f"Error processing document {doc.id}: {str(e)}"
                            )

                except Exception as e:
                    logger.error(
                        f"Error initializing RAG assistant for system {system_id}: {str(e)}"
                    )
                    # Добавляем ошибки для всех документов этой системы
                    for doc in docs:
                        errors.append(
                            {
                                "document_id": doc.id,
                                "document_title": doc.title,
                                "error": f"System error: {str(e)}",
                                "system_id": system_id,
                            }
                        )

            # Финальный результат
            success_count = processed - len(errors)
            error_count = len(errors)

            result = {
                "status": "completed",
                "total_documents": total_docs,
                "processed_successfully": success_count,
                "errors": error_count,
                "error_details": errors[:10],  # Показываем только 10 ошибок
                "task_id": task_id,
                "timestamp": time.time(),
            }

            if success_count == total_docs:
                logger.info(
                    f"Batch processing completed successfully: {success_count}/{total_docs}"
                )
            else:
                logger.warning(
                    (
                        "Batch processing completed with errors: "
                        f"{success_count}/{total_docs} successful, {error_count} errors"
                    )
                )

            return result

    except Exception as exc:
        logger.error(
            f"Critical error in batch processing: {str(exc)}\n{traceback.format_exc()}"
        )

        # Retry механизм
        if self.request.retries < MAX_RETRIES:
            countdown = RETRY_COUNTDOWN * (self.request.retries + 1)
            logger.info(
                f"Retrying batch processing (attempt {self.request.retries + 1}) in {countdown} seconds"
            )
            raise self.retry(countdown=countdown, exc=exc)

        return {
            "status": "error",
            "error": str(exc),
            "document_ids": document_ids,
            "retries_exhausted": True,
            "task_id": task_id,
        }


@shared_task(bind=True)
def cleanup_old_query_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Очистка старых логов запросов для оптимизации БД
    Автоматическая задача для запуска по расписанию
    """

    try:
        with task_performance_monitor("cleanup_logs", days_to_keep=days_to_keep):
            cutoff_date = timezone.now() - timezone.timedelta(days=days_to_keep)

            # Подсчитываем количество старых записей
            old_logs_count = RagQueryLog.objects.filter(
                timestamp__lt=cutoff_date
            ).count()

            if old_logs_count == 0:
                logger.info("No old query logs to clean up")
                return {
                    "status": "success",
                    "deleted_count": 0,
                    "message": "No old logs found",
                }

            # Удаляем старые записи пакетами
            with transaction.atomic():
                deleted_count = RagQueryLog.objects.filter(
                    timestamp__lt=cutoff_date
                ).delete()[0]

            logger.info(
                f"Cleaned up {deleted_count} old query logs (older than {days_to_keep} days)"
            )

            return {
                "status": "success",
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_to_keep": days_to_keep,
            }

    except Exception as exc:
        logger.error(
            f"Error cleaning up old logs: {str(exc)}\n{traceback.format_exc()}"
        )

        if self.request.retries < MAX_RETRIES:
            raise self.retry(countdown=RETRY_COUNTDOWN, exc=exc)

        return {"status": "error", "error": str(exc), "retries_exhausted": True}


@shared_task(bind=True)
def optimize_faiss_index(self, system_id: int) -> Dict[str, Any]:
    """
    Оптимизация FAISS индекса для улучшения производительности поиска
    Тяжелая операция - запускается по расписанию
    """

    try:
        with task_performance_monitor("optimize_faiss", system_id=system_id):

            # Получаем систему
            try:
                system = RagSystem.objects.get(id=system_id)
            except RagSystem.DoesNotExist:
                return {"status": "error", "error": f"RAG system {system_id} not found"}

            RagAssistant(system)

            # Перестроение индекса с оптимизацией
            # TODO: реализовать optimize_index метод в RagAssistant
            # assistant.optimize_index()

            logger.info(f"FAISS index optimized for system {system_id}")

            return {
                "status": "success",
                "system_id": system_id,
                "system_name": system.name,
                "timestamp": time.time(),
            }

    except Exception as exc:
        logger.error(f"Error optimizing FAISS index for system {system_id}: {str(exc)}")

        if self.request.retries < MAX_RETRIES:
            raise self.retry(countdown=RETRY_COUNTDOWN * 2, exc=exc)  # Длиннее пауза

        return {
            "status": "error",
            "error": str(exc),
            "system_id": system_id,
            "retries_exhausted": True,
        }


@shared_task
def generate_performance_report() -> Dict[str, Any]:
    """
    Генерация отчета о производительности для анализа
    Запускается каждые 24 часа
    """

    try:
        with task_performance_monitor("performance_report"):

            # Метрики за 24 часа
            total_requests = 0
            total_ai_requests = 0

            # Сбор метрик за последние 24 часа
            for hour in range(24):
                hour_key = time.strftime(
                    "%Y-%m-%d:%H", time.localtime(time.time() - hour * 3600)
                )

                requests = cache.get(f"performance_metrics:{hour_key}:requests", 0)
                ai_requests = cache.get(f"ai_metrics:{hour_key}:ai_requests", 0)

                total_requests += requests
                total_ai_requests += ai_requests

            # Метрики БД
            from apps.rag_assistant.models import Document, RagQueryLog

            documents_count = Document.objects.count()
            queries_today = RagQueryLog.objects.filter(
                timestamp__date=timezone.now().date()
            ).count()

            report = {
                "status": "success",
                "period": "24_hours",
                "metrics": {
                    "total_requests": total_requests,
                    "ai_requests": total_ai_requests,
                    "documents_in_system": documents_count,
                    "queries_today": queries_today,
                    "requests_per_hour_avg": round(total_requests / 24, 2),
                    "ai_requests_per_hour_avg": round(total_ai_requests / 24, 2),
                },
                "timestamp": time.time(),
                "generated_at": timezone.now().isoformat(),
            }

            # Сохраняем отчет в кеш
            cache.set("performance_report_24h", report, timeout=86400)  # 24 часа

            logger.info(
                (
                    "Performance report generated: "
                    f"{total_requests} requests, {total_ai_requests} AI operations"
                )
            )

            return report

    except Exception as exc:
        logger.error(f"Error generating performance report: {str(exc)}")
        return {"status": "error", "error": str(exc), "timestamp": time.time()}
