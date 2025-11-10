"""Сервисный модуль RAG Assistant с обновленными валидаторами Pydantic V2."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
import tempfile
from typing import Any

import bleach  # type: ignore[import-untyped]
import numpy as np
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, field_validator

from .llm_factory import LLMFactory
from .models import Document, RagQueryLog, RagSystem

# Константы
MAX_QUERY_LENGTH = 500
MAX_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB

CACHE_VERSION = "v1"
DOC_EMBEDDING_TTL: int | None = None
SEARCH_RESULT_TTL: int = 3600
FAQ_ANSWER_TTL: int = 86400
CACHE_STATS_KEY = "cache_stats"

LOADER_MAP: dict[str, Callable[[str], Any]] = {}


class QueryInput(BaseModel):
    """Модель ввода для RAG запросов с валидацией."""

    query: str
    user_id: int | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        """Валидирует и очищает входной запрос.

        Args:
            v: Входное значение запроса

        Returns:
            Очищенный и валидированный запрос

        Raises:
            ValueError: Если запрос не строка или слишком длинный
        """
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long. Maximum length is {MAX_QUERY_LENGTH} characters"
            )
        sanitized = bleach.clean(v, tags=[], attributes={}, strip=True)
        return sanitized.strip()

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: Any) -> int | None:
        """Валидирует ID пользователя.

        Args:
            v: Входное значение ID пользователя

        Returns:
            Валидированный ID пользователя или None

        Raises:
            ValueError: Если ID пользователя не целое число
        """
        if v is not None and not isinstance(v, int):
            raise ValueError("User ID must be an integer")
        return v


class DocumentInput(BaseModel):
    """Модель ввода для документов с валидацией."""

    content: str
    format: str
    metadata: dict[str, Any]

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        """Валидирует и очищает содержимое документа.

        Args:
            v: Входное значение содержимого

        Returns:
            Очищенное и валидированное содержимое

        Raises:
            ValueError: Если содержимое не строка или слишком большое
        """
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        if len(v.encode("utf-8")) > MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content too large. Maximum size is {MAX_CONTENT_SIZE} bytes"
            )
        return bleach.clean(v, tags=[], attributes={}, strip=True)

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: Any) -> str:
        """Валидирует формат документа.

        Args:
            v: Входное значение формата

        Returns:
            Валидированный формат

        Raises:
            ValueError: Если формат не строка
        """
        if not isinstance(v, str):
            raise ValueError("Format must be a string")
        return v.strip().lower()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Any) -> dict[str, Any]:
        """Валидирует и очищает метаданные.

        Args:
            v: Входное значение метаданных

        Returns:
            Очищенные метаданные

        Raises:
            ValueError: Если метаданные не словарь
        """
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        sanitized_metadata: dict[str, Any] = {}
        for key, value in v.items():
            if isinstance(value, str):
                sanitized_metadata[key] = bleach.clean(
                    value, tags=[], attributes={}, strip=True
                )
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata


class CacheStats:
    """Управление статистикой кэша."""

    @staticmethod
    def increment_hit() -> None:
        """Увеличивает счетчик попаданий в кэш."""
        stats: dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["hits"] = stats.get("hits", 0) + 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def increment_miss() -> None:
        """Увеличивает счетчик промахов кэша."""
        stats: dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["misses"] = stats.get("misses", 0) + 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def get_stats() -> dict[str, int]:
        """Возвращает статистику кэша.

        Returns:
            Словарь с количеством попаданий и промахов
        """
        stats: dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        return stats

    @staticmethod
    def get_hit_rate() -> float:
        """Вычисляет процент попаданий в кэш.

        Returns:
            Процент попаданий от 0.0 до 1.0
        """
        stats = CacheStats.get_stats()
        total = stats.get("hits", 0) + stats.get("misses", 0)
        return (stats.get("hits", 0) / total) if total > 0 else 0.0


class RagAssistant:
    """RAG (Retrieval-Augmented Generation) Assistant."""

    def __init__(self, system: RagSystem):
        """Инициализирует RAG Assistant с конфигурацией системы.

        Args:
            system: Конфигурация RAG системы

        Raises:
            TypeError: Если system не экземпляр RagSystem
        """
        if not isinstance(system, RagSystem):
            raise TypeError("system must be an instance of RagSystem")

        self.system = system
        self.llm = LLMFactory.create_chat_model()
        self.embedder = LLMFactory.create_embedder()

    def _get_cache_key(
        self, key_type: str, identifier: str, version: str = CACHE_VERSION
    ) -> str:
        """Генерирует ключ кэша для хранения/получения данных.

        Args:
            key_type: Тип ключа (faq, search, etc.)
            identifier: Идентификатор данных
            version: Версия кэша

        Returns:
            Сгенерированный ключ кэша
        """
        sys_id = getattr(self.system, "pk", None)
        return f"rag:{sys_id}:{key_type}:{identifier}:{version}"

    def _cache_faq_answer(self, question: str, answer: str) -> None:
        """Кэширует ответ FAQ для будущего использования.

        Args:
            question: Вопрос
            answer: Ответ
        """
        cache_key = self._get_cache_key(
            "faq", hashlib.sha256(question.encode()).hexdigest()
        )
        cache.set(cache_key, answer, timeout=FAQ_ANSWER_TTL)

    def _get_cached_faq_answer(self, question: str) -> str | None:
        """Получает кэшированный ответ FAQ если существует.

        Args:
            question: Вопрос

        Returns:
            Кэшированный ответ или None
        """
        cache_key = self._get_cache_key(
            "faq", hashlib.sha256(question.encode()).hexdigest()
        )
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return str(cached)
        CacheStats.increment_miss()
        return None

    def _build_retriever(self) -> Callable[[str], list[dict[str, Any]]]:
        """Строит функцию извлечения документов.

        Returns:
            Функция извлечения документов
        """
        from .rag_core import default_local_orchestrator

        orchestrator = default_local_orchestrator()
        vindex = orchestrator.load_index(
            getattr(self.system, "index_version", None) or "test_v1"
        )
        docs_list: list[str] = (getattr(vindex, "metadata", None) or {}).get("docs", [])

        def _retrieve(question: str) -> list[dict[str, Any]]:
            """Извлекает релевантные документы для вопроса.

            Args:
                question: Вопрос пользователя

            Returns:
                Список релевантных документов
            """
            q_emb = self._encode([question])
            _, indices = vindex.search(q_emb, k=4)
            results: list[dict[str, Any]] = []
            for i in indices[0]:
                idx = int(i)
                if 0 <= idx < len(docs_list):
                    results.append({"content": docs_list[idx], "meta": {"idx": idx}})
            return results

        return _retrieve

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Кодирует тексты в эмбеддинги.

        Args:
            texts: Список текстов для кодирования

        Returns:
            Массив нормализованных эмбеддингов
        """
        emb = self.embedder.embed_documents(texts)
        arr = np.array(emb, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")

    def _build_chain(self) -> Any:
        """Строит цепочку обработки RAG.

        Returns:
            Собранная цепочка RAG
        """
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant for hydraulic diagnostics.
            Use the following retrieved context to answer the user's question concisely and accurately.
            If the answer is not present in the context, say you don't know.

            Context:
            {context}

            Question:
            {question}

            Answer in the same language as the question.
            """
        )
        parser = StrOutputParser()

        retriever_fn = self._build_retriever()

        def format_docs(docs: list[dict[str, Any]]) -> str:
            """Форматирует извлеченные документы для контекста.

            Args:
                docs: Список извлеченных документов

            Returns:
                Отформатированный контекст
            """
            return "\n\n".join(str(d["content"]) for d in docs if d.get("content"))

        return (
            {"question": RunnablePassthrough()}
            | {"docs": RunnableLambda(lambda x: retriever_fn(x["question"]))}
            | RunnableLambda(
                lambda x: {"question": x["question"], "context": format_docs(x["docs"])}
            )
            | prompt
            | self.llm
            | parser
        )

    def answer(self, query: str, user_id: int | None = None) -> str:
        """Генерирует ответ на пользовательский запрос используя RAG.

        Args:
            query: Запрос пользователя
            user_id: ID пользователя (опционально)

        Returns:
            Сгенерированный ответ

        Raises:
            ValidationError: Если валидация запроса не удалась
        """
        cached_answer = self._get_cached_faq_answer(query)
        if cached_answer is not None:
            self._log_query(query, cached_answer, user_id)
            return cached_answer

        try:
            validated_input = QueryInput(query=query, user_id=user_id)
        except Exception as e:
            raise ValidationError(f"Query validation failed: {e}")

        chain = self._build_chain()
        result: str = str(chain.invoke({"question": validated_input.query}))

        self._cache_faq_answer(validated_input.query, result)
        self._log_query(validated_input.query, result, validated_input.user_id)
        return result

    @transaction.atomic
    def _log_query(self, query: str, response: str, user_id: int | None = None) -> None:
        """Логирует запрос и ответ в базу данных.

        Args:
            query: Запрос пользователя
            response: Ответ системы
            user_id: ID пользователя (опционально)
        """
        RagQueryLog.objects.create(
            system=self.system,
            query_text=query,
            response_text=response,
            user_id=user_id,
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Возвращает статистику производительности кэша.

        Returns:
            Словарь со статистикой кэша
        """
        stats = CacheStats.get_stats()
        hit_rate = CacheStats.get_hit_rate()
        return {
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate": hit_rate,
            "hit_rate_percent": round(hit_rate * 100, 2),
        }

    def clear_cache_stats(self) -> None:
        """Очищает статистику кэша."""
        cache.delete(CACHE_STATS_KEY)


def tmp_file_for(doc: Document) -> str:
    """Создает временный файл для обработки документа.

    Args:
        doc: Экземпляр документа для создания файла

    Returns:
        Путь к созданному временному файлу

    Raises:
        TypeError: Если doc не экземпляр Document
        ValueError: Если формат или содержимое документа отсутствуют/невалидны
    """
    if not isinstance(doc, Document):
        raise TypeError("doc must be an instance of Document")

    if not getattr(doc, "format", None):
        raise ValueError("Document format is required")

    if not getattr(doc, "content", None):
        raise ValueError("Document content is required")

    suffix = f".{doc.format}"
    fd, path = tempfile.mkstemp(suffix=suffix)

    content_bytes = doc.content.encode("utf-8")
    if len(content_bytes) > MAX_CONTENT_SIZE:
        os.close(fd)
        os.unlink(path)
        raise ValueError(f"Content too large. Maximum size is {MAX_CONTENT_SIZE} bytes")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            sanitized_content = bleach.clean(
                doc.content, tags=[], attributes={}, strip=True
            )
            f.write(sanitized_content)
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise

    return path
