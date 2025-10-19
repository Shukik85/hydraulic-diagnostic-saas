from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any, Dict, List, Optional

from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction

import bleach  # type: ignore[import-untyped]
import pydantic
from django_ratelimit.decorators import ratelimit
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .llm_factory import LLMFactory
from .models import Document, RagQueryLog, RagSystem

# Константы для валидации
MAX_QUERY_LENGTH = 500
MAX_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB

# Константы для кеширования
CACHE_VERSION = "v1"
DOC_EMBEDDING_TTL = None  # Бессрочно
SEARCH_RESULT_TTL = 3600  # 1 час
FAQ_ANSWER_TTL = 86400  # 24 часа
CACHE_STATS_KEY = "cache_stats"

LOADER_MAP = {
    # В текущей миграции оставляем только текстовые форматы,
    # при необходимости подключим unstructured/pdfminer позднее.
}


class QueryInput(pydantic.BaseModel):
    """Pydantic модель для валидации входного запроса"""

    query: str
    user_id: Optional[int] = None

    @pydantic.validator("query")
    def validate_query(cls, v):
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long. Maximum length is {MAX_QUERY_LENGTH} characters"
            )
        # Санитизация HTML-тегов
        sanitized = bleach.clean(v, tags=[], attributes={}, strip=True)
        return sanitized.strip()

    @pydantic.validator("user_id")
    def validate_user_id(cls, v):
        if v is not None and not isinstance(v, int):
            raise ValueError("User ID must be an integer")
        return v


class DocumentInput(pydantic.BaseModel):
    """Pydantic модель для валидации входных данных документа"""

    content: str
    format: str
    metadata: dict

    @pydantic.validator("content")
    def validate_content(cls, v):
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        if len(v.encode("utf-8")) > MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content too large. Maximum size is {MAX_CONTENT_SIZE} bytes"
            )
        # Санитизация HTML-тегов
        return bleach.clean(v, tags=[], attributes={}, strip=True)

    @pydantic.validator("format")
    def validate_format(cls, v):
        if not isinstance(v, str):
            raise ValueError("Format must be a string")
        return v.strip().lower()

    @pydantic.validator("metadata")
    def validate_metadata(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        # Санитизация всех строковых значений в metadata
        sanitized_metadata = {}
        for key, value in v.items():
            if isinstance(value, str):
                sanitized_metadata[key] = bleach.clean(
                    value, tags=[], attributes={}, strip=True
                )
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata


class CacheStats:
    """Класс для отслеживания статистики кеширования"""

    @staticmethod
    def increment_hit():
        stats = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["hits"] += 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def increment_miss():
        stats = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["misses"] += 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def get_stats():
        return cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})

    @staticmethod
    def get_hit_rate():
        stats = CacheStats.get_stats()
        total = stats["hits"] + stats["misses"]
        return stats["hits"] / total if total > 0 else 0


class RagAssistant:
    def __init__(self, system: RagSystem):
        if not isinstance(system, RagSystem):
            raise TypeError("system must be an instance of RagSystem")

        self.system = system
        # LLM и эмбеддинги через Ollama по умолчанию (переключаемо через settings)
        self.llm = LLMFactory.create_chat_model()
        self.embedder = LLMFactory.create_embedder()

    def _get_cache_key(
        self, key_type: str, identifier: str, version: str = CACHE_VERSION
    ) -> str:
        return f"rag:{self.system.id}:{key_type}:{identifier}:{version}"

    def _cache_faq_answer(self, question: str, answer: str):
        cache_key = self._get_cache_key(
            "faq", hashlib.md5(question.encode()).hexdigest()
        )
        cache.set(cache_key, answer, timeout=FAQ_ANSWER_TTL)

    def _get_cached_faq_answer(self, question: str) -> Optional[str]:
        cache_key = self._get_cache_key(
            "faq", hashlib.md5(question.encode()).hexdigest()
        )
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return cached
        CacheStats.increment_miss()
        return None

    # --------- Retrieval адаптер над вашим FAISS индексом из rag_core --------- #

    def _build_retriever(self):
        # Предполагается, что индекс и метаданные (docs) доступны через rag_core orchestrator
        # Здесь упрощённо: получаем из системы путь/версию индекса
        from .rag_core import default_local_orchestrator

        orchestrator = default_local_orchestrator()
        vindex = orchestrator.load_index(self.system.index_version or "test_v1")
        docs_list = (vindex.metadata or {}).get("docs", [])

        def _retrieve(question: str):
            q_emb = self._encode([question])
            _, indices = vindex.search(q_emb, k=4)
            results = []
            for i in indices[0]:
                if 0 <= i < len(docs_list):
                    results.append({"content": docs_list[i], "meta": {"idx": int(i)}})
            return results

        return _retrieve

    def _encode(self, texts: List[str]):
        # OllamaEmbeddings интерфейс возвращает список векторов
        emb = self.embedder.embed_documents(texts)
        # нормализуем для inner product (опционально)
        import numpy as np

        arr = np.array(emb, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")

    # -------------------------- LangChain 1.x chain -------------------------- #

    def _build_chain(self):
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

        def format_docs(docs: List[Dict[str, Any]]) -> str:
            return "\n\n".join(d["content"] for d in docs if d.get("content"))

        chain = (
            {"question": RunnablePassthrough()}
            | {"docs": RunnableLambda(lambda x: retriever_fn(x["question"]))}
            | RunnableLambda(lambda x: {"question": x["question"], "context": format_docs(x["docs"])})
            | prompt
            | self.llm
            | parser
        )
        return chain

    @ratelimit(key="user_or_ip", rate="10/m", block=True)
    def answer(self, query: str, user_id: Optional[int] = None):
        cached_answer = self._get_cached_faq_answer(query)
        if cached_answer is not None:
            self._log_query(query, cached_answer, user_id)
            return cached_answer

        try:
            validated_input = QueryInput(query=query, user_id=user_id)
        except pydantic.ValidationError as e:
            raise ValidationError(f"Query validation failed: {e}")

        chain = self._build_chain()
        result: str = chain.invoke({"question": validated_input.query})

        self._cache_faq_answer(validated_input.query, result)
        self._log_query(validated_input.query, result, validated_input.user_id)
        return result

    @transaction.atomic
    def _log_query(self, query: str, response: str, user_id: Optional[int] = None):
        RagQueryLog.objects.create(
            system=self.system,
            query_text=query,
            response_text=response,
            user_id=user_id,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        stats = CacheStats.get_stats()
        hit_rate = CacheStats.get_hit_rate()
        return {
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": hit_rate,
            "hit_rate_percent": round(hit_rate * 100, 2),
        }

    def clear_cache_stats(self):
        cache.delete(CACHE_STATS_KEY)


def tmp_file_for(doc: Document):
    if not isinstance(doc, Document):
        raise TypeError("doc must be an instance of Document")

    if not hasattr(doc, "format") or not doc.format:
        raise ValueError("Document format is required")

    if not hasattr(doc, "content") or not doc.content:
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
    except Exception as e:
        if os.path.exists(path):
            os.unlink(path)
        raise e

    return path
