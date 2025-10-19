import hashlib
import os
import tempfile
from typing import Any, Dict, List, Optional

import bleach  # type: ignore[import-untyped]
import pydantic
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from django_ratelimit.decorators import ratelimit
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    MarkdownLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

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
    "txt": TextLoader,
    "pdf": PDFMinerLoader,
    "docx": UnstructuredWordDocumentLoader,
    "md": MarkdownLoader,
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
            raise ValueError(f"Query too long. Maximum length is {MAX_QUERY_LENGTH} characters")
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
            raise ValueError(f"Content too large. Maximum size is {MAX_CONTENT_SIZE} bytes")
        # Санитизация HTML-тегов
        return bleach.clean(v, tags=[], attributes={}, strip=True)

    @pydantic.validator("format")
    def validate_format(cls, v):
        if not isinstance(v, str):
            raise ValueError("Format must be a string")
        if v not in LOADER_MAP:
            raise ValueError(f"Unsupported format. Supported formats: {list(LOADER_MAP.keys())}")
        return v.strip().lower()

    @pydantic.validator("metadata")
    def validate_metadata(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        # Санитизация всех строковых значений в metadata
        sanitized_metadata = {}
        for key, value in v.items():
            if isinstance(value, str):
                sanitized_metadata[key] = bleach.clean(value, tags=[], attributes={}, strip=True)
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
        self.embedding = OpenAIEmbeddings()
        self.index = self._load_index()

    def _load_index(self):
        """Загрузка индекса с валидацией"""
        if self.system.index_config is not None and not isinstance(self.system.index_config, dict):
            raise TypeError("index_config must be a dictionary")

        config = self.system.index_config or {}
        if self.system.index_type == "faiss":
            path = config.get("index_path") or f"/tmp/{self.system.name}.faiss"
            if not isinstance(path, str):
                raise TypeError("index_path must be a string")

            if os.path.exists(path):
                return FAISS.load_local(path, self.embedding)
            return FAISS(self.embedding.embed_query, [])
        raise NotImplementedError("Only FAISS index type is currently supported")

    def _get_cache_key(self, key_type: str, identifier: str, version: str = CACHE_VERSION) -> str:
        return f"rag:{self.system.id}:{key_type}:{identifier}:{version}"

    def _invalidate_document_cache(self, doc_id: int):
        cache_keys_to_delete = [
            self._get_cache_key("doc_embedding", str(doc_id)),
            self._get_cache_key("doc_chunks", str(doc_id)),
        ]
        cache.delete_many(cache_keys_to_delete)

    def _cache_document_embedding(self, doc_id: int, embedding: List[float]):
        cache_key = self._get_cache_key("doc_embedding", str(doc_id))
        cache.set(cache_key, embedding, timeout=DOC_EMBEDDING_TTL)

    def _get_cached_document_embedding(self, doc_id: int) -> Optional[List[float]]:
        cache_key = self._get_cache_key("doc_embedding", str(doc_id))
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return cached
        CacheStats.increment_miss()
        return None

    def _cache_search_result(self, query: str, category: Optional[str], results: List[Dict]):
        cache_key = self._get_cache_key(
            "search", f"{hashlib.md5(query.encode()).hexdigest()}:{category}"
        )
        cache.set(cache_key, results, timeout=SEARCH_RESULT_TTL)

    def _get_cached_search_result(
        self, query: str, category: Optional[str]
    ) -> Optional[List[Dict]]:
        cache_key = self._get_cache_key(
            "search", f"{hashlib.md5(query.encode()).hexdigest()}:{category}"
        )
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return cached
        CacheStats.increment_miss()
        return None

    def _cache_faq_answer(self, question: str, answer: str):
        cache_key = self._get_cache_key("faq", hashlib.md5(question.encode()).hexdigest())
        cache.set(cache_key, answer, timeout=FAQ_ANSWER_TTL)

    def _get_cached_faq_answer(self, question: str) -> Optional[str]:
        cache_key = self._get_cache_key("faq", hashlib.md5(question.encode()).hexdigest())
        cached = cache.get(cache_key)
        if cached is not None:
            CacheStats.increment_hit()
            return cached
        CacheStats.increment_miss()
        return None

    @transaction.atomic
    def index_document(self, doc: Document):
        try:
            validated_doc = DocumentInput(
                content=doc.content, format=doc.format, metadata=doc.metadata or {}
            )
        except pydantic.ValidationError as e:
            raise ValidationError(f"Document validation failed: {e}")

        doc.content = validated_doc.content
        doc.format = validated_doc.format
        doc.metadata = validated_doc.metadata

        self._invalidate_document_cache(doc.id)

        loader_cls = LOADER_MAP.get(doc.format)
        if loader_cls is None:
            raise ValueError(f"Unsupported document format: {doc.format}")

        loader = loader_cls(file_path=doc.metadata.get("file_path") or tmp_file_for(doc))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        self.index.add_documents(chunks)
        self.index.save_local(f"/tmp/{self.system.name}.faiss")

    def search_documents(
        self, query: str, category: Optional[str] = None, ttl: int = 3600
    ) -> List[Dict]:
        cached_results = self._get_cached_search_result(query, category)
        if cached_results is not None:
            return cached_results

        results = self._perform_search(query, category)
        self._cache_search_result(query, category, results)
        return results

    def _perform_search(self, query: str, category: Optional[str] = None) -> List[Dict]:
        query_embedding = self._get_or_generate_embedding(query)
        docs = self.index.similarity_search_by_vector(query_embedding, k=10)
        results: List[Dict[str, Any]] = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "similarity", None),
                }
            )
        return results

    def _get_or_generate_embedding(self, text: str) -> List[float]:
        return self.embedding.embed_query(text)

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

        validated_query = validated_input.query
        validated_user_id = validated_input.user_id

        qa = RetrievalQA.from_chain_type(
            llm=self.system.model_name,
            chain_type="stuff",
            retriever=self.index.as_retriever(),
        )

        result = qa.run(validated_query)
        self._cache_faq_answer(validated_query, result)
        self._log_query(validated_query, result, validated_user_id)
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
            sanitized_content = bleach.clean(doc.content, tags=[], attributes={}, strip=True)
            f.write(sanitized_content)
    except Exception as e:
        if os.path.exists(path):
            os.unlink(path)
        raise e

    return path
