from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction

import bleach  # type: ignore[import-untyped]
import pydantic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .llm_factory import LLMFactory
from .models import Document, RagQueryLog, RagSystem

MAX_QUERY_LENGTH = 500
MAX_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB

CACHE_VERSION = "v1"
DOC_EMBEDDING_TTL: Optional[int] = None
SEARCH_RESULT_TTL: int = 3600
FAQ_ANSWER_TTL: int = 86400
CACHE_STATS_KEY = "cache_stats"

LOADER_MAP: Dict[str, Callable[[str], Any]] = {}


class QueryInput(pydantic.BaseModel):
    query: str
    user_id: Optional[int] = None

    @pydantic.validator("query")
    def validate_query(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Query must be a string")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long. Maximum length is {MAX_QUERY_LENGTH} characters"
            )
        sanitized = bleach.clean(v, tags=[], attributes={}, strip=True)
        return sanitized.strip()

    @pydantic.validator("user_id")
    def validate_user_id(cls, v: Any) -> Optional[int]:
        if v is not None and not isinstance(v, int):
            raise ValueError("User ID must be an integer")
        return v


class DocumentInput(pydantic.BaseModel):
    content: str
    format: str
    metadata: Dict[str, Any]

    @pydantic.validator("content")
    def validate_content(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        if len(v.encode("utf-8")) > MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content too large. Maximum size is {MAX_CONTENT_SIZE} bytes"
            )
        return bleach.clean(v, tags=[], attributes={}, strip=True)

    @pydantic.validator("format")
    def validate_format(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Format must be a string")
        return v.strip().lower()

    @pydantic.validator("metadata")
    def validate_metadata(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        sanitized_metadata: Dict[str, Any] = {}
        for key, value in v.items():
            if isinstance(value, str):
                sanitized_metadata[key] = bleach.clean(
                    value, tags=[], attributes={}, strip=True
                )
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata


class CacheStats:
    @staticmethod
    def increment_hit() -> None:
        stats: Dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["hits"] = stats.get("hits", 0) + 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def increment_miss() -> None:
        stats: Dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        stats["misses"] = stats.get("misses", 0) + 1
        cache.set(CACHE_STATS_KEY, stats)

    @staticmethod
    def get_stats() -> Dict[str, int]:
        stats: Dict[str, int] = cache.get(CACHE_STATS_KEY, {"hits": 0, "misses": 0})
        return stats

    @staticmethod
    def get_hit_rate() -> float:
        stats = CacheStats.get_stats()
        total = stats.get("hits", 0) + stats.get("misses", 0)
        return (stats.get("hits", 0) / total) if total > 0 else 0.0


class RagAssistant:
    def __init__(self, system: RagSystem):
        if not isinstance(system, RagSystem):
            raise TypeError("system must be an instance of RagSystem")

        self.system = system
        self.llm = LLMFactory.create_chat_model()
        self.embedder = LLMFactory.create_embedder()

    def _get_cache_key(self, key_type: str, identifier: str, version: str = CACHE_VERSION) -> str:
        return f"rag:{self.system.id}:{key_type}:{identifier}:{version}"

    def _cache_faq_answer(self, question: str, answer: str) -> None:
        cache_key = self._get_cache_key("faq", hashlib.md5(question.encode()).hexdigest())
        cache.set(cache_key, answer, timeout=FAQ_ANSWER_TTL)

    def _get_cached_faq_answer(self, question: str) -> Optional[str]:
        cache_key = self._get_cache_key("faq", hashlib.md5(question.encode()).hexdigest())
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return str(cached)
        CacheStats.increment_miss()
        return None

    def _build_retriever(self) -> Callable[[str], List[Dict[str, Any]]]:
        from .rag_core import default_local_orchestrator

        orchestrator = default_local_orchestrator()
        vindex = orchestrator.load_index(getattr(self.system, "index_version", None) or "test_v1")
        docs_list: List[str] = (getattr(vindex, "metadata", None) or {}).get("docs", [])

        def _retrieve(question: str) -> List[Dict[str, Any]]:
            q_emb = self._encode([question])
            _, indices = vindex.search(q_emb, k=4)
            results: List[Dict[str, Any]] = []
            for i in indices[0]:
                idx = int(i)
                if 0 <= idx < len(docs_list):
                    results.append({"content": docs_list[idx], "meta": {"idx": idx}})
            return results

        return _retrieve

    def _encode(self, texts: List[str]):
        emb = self.embedder.embed_documents(texts)
        import numpy as np

        arr = np.array(emb, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype("float32")

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
            return "\n\n".join(str(d["content"]) for d in docs if d.get("content"))

        chain = (
            {"question": RunnablePassthrough()}
            | {"docs": RunnableLambda(lambda x: retriever_fn(x["question"]))}
            | RunnableLambda(lambda x: {"question": x["question"], "context": format_docs(x["docs"])})
            | prompt
            | self.llm
            | parser
        )
        return chain

    def answer(self, query: str, user_id: Optional[int] = None) -> str:
        cached_answer = self._get_cached_faq_answer(query)
        if cached_answer is not None:
            self._log_query(query, cached_answer, user_id)
            return cached_answer

        try:
            validated_input = QueryInput(query=query, user_id=user_id)
        except pydantic.ValidationError as e:
            raise ValidationError(f"Query validation failed: {e}")

        chain = self._build_chain()
        result: str = str(chain.invoke({"question": validated_input.query}))

        self._cache_faq_answer(validated_input.query, result)
        self._log_query(validated_input.query, result, validated_input.user_id)
        return result

    @transaction.atomic
    def _log_query(self, query: str, response: str, user_id: Optional[int] = None) -> None:
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
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate": hit_rate,
            "hit_rate_percent": round(hit_rate * 100, 2),
        }

    def clear_cache_stats(self) -> None:
        cache.delete(CACHE_STATS_KEY)


def tmp_file_for(doc: Document) -> str:
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
            sanitized_content = bleach.clean(doc.content, tags=[], attributes={}, strip=True)
            f.write(sanitized_content)
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise

    return path
