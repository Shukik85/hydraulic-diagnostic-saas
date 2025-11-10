"""Основной модуль RAG системы - абстракции и локальное хранилище."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------- Слой хранилища ---------------------------- #


class StorageBackend(Protocol):
    """Протокол для бэкендов хранения индексов."""

    def save_index(
        self, version: str, index_bytes: bytes, metadata: dict[str, Any]
    ) -> str:
        """Сохраняет индекс и метаданные.

        Args:
            version: Версия индекса
            index_bytes: Байты индекса FAISS
            metadata: Метаданные индекса

        Returns:
            Путь к сохраненному индексу
        """
        ...

    def load_index(self, version: str) -> tuple[bytes, dict[str, Any]]:
        """Загружает индекс и метаданные.

        Args:
            version: Версия индекса

        Returns:
            Кортеж (байты индекса, метаданные)
        """
        ...

    def list_versions(self) -> list[str]:
        """Возвращает список доступных версий индексов."""
        ...


@dataclass
class LocalStorageBackend:
    """Локальное хранилище индексов на файловой системе."""

    base_path: Path

    def __post_init__(self) -> None:
        """Создает базовый путь при инициализации."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        """Возвращает путь к директории версии.

        Args:
            version: Версия индекса

        Returns:
            Путь к директории версии
        """
        d = self.base_path / f"v_{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_index(
        self, version: str, index_bytes: bytes, metadata: dict[str, Any]
    ) -> str:
        """Сохраняет индекс и метаданные атомарно.

        Args:
            version: Версия индекса
            index_bytes: Байты индекса FAISS
            metadata: Метаданные индекса

        Returns:
            Путь к сохраненному индексу
        """
        vdir = self._version_dir(version)
        index_path = vdir / "index.faiss"
        meta_path = vdir / "metadata.json"

        # Атомарная запись через временные файлы
        tmp_index = vdir / "index.faiss.tmp"
        with open(tmp_index, "wb") as f:
            f.write(index_bytes)
        tmp_index.replace(index_path)

        tmp_meta = vdir / "metadata.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        tmp_meta.replace(meta_path)
        return str(vdir)

    def load_index(self, version: str) -> tuple[bytes, dict[str, Any]]:
        """Загружает индекс и метаданные.

        Args:
            version: Версия индекса

        Returns:
            Кортеж (байты индекса, метаданные)

        Raises:
            FileNotFoundError: Если индекс или метаданные не найдены
        """
        vdir = self._version_dir(version)
        index_path = vdir / "index.faiss"
        meta_path = vdir / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index or metadata not found for version {version}"
            )
        with open(index_path, "rb") as f:
            idx_bytes = f.read()
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return idx_bytes, metadata

    def list_versions(self) -> list[str]:
        """Возвращает список доступных версий индексов.

        Returns:
            Отсортированный список версий
        """
        versions = []
        for p in self.base_path.glob("v_*"):
            if p.is_dir():
                versions.append(p.name.removeprefix("v_"))
        versions.sort()
        return versions


# ------------------------- Провайдер эмбеддингов ------------------------- #


@dataclass
class EmbeddingsProvider:
    """Провайдер эмбеддингов на основе SentenceTransformers."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None  # например, "cpu" | "cuda"

    _model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Инициализирует модель при необходимости.

        Returns:
            Инициализированная модель SentenceTransformer
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self, texts: list[str], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        """Кодирует тексты в эмбеддинги.

        Args:
            texts: Список текстов для кодирования
            batch_size: Размер батча для кодирования
            normalize: Нормализовать ли векторы (L2 норма)

        Returns:
            Массив эмбеддингов размерности (n_texts, embedding_dim)
        """
        model = self._ensure_model()
        embeddings = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        if normalize:
            # L2 нормализация для косинусного сходства с inner product индексом
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings.astype("float32")


# ------------------------------ Векторный индекс --------------------------- #


@dataclass
class VectorIndex:
    """Векторный индекс на основе FAISS."""

    dim: int
    metric: str = "ip"  # inner product (косинусное сходство при L2 нормализации)
    _index: faiss.Index | None = None
    metadata: dict[str, Any] | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Строит индекс из векторов.

        Args:
            vectors: Массив векторов размерности (n_vectors, dim)

        Raises:
            AssertionError: Если размерности не совпадают
            ValueError: Если метрика не поддерживается
        """
        assert vectors.ndim == 2
        assert vectors.shape[1] == self.dim
        if self.metric == "ip":
            index = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError("Unsupported metric: " + self.metric)
        index.add(vectors)
        self._index = index

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Выполняет поиск ближайших соседей.

        Args:
            queries: Запросы размерности (n_queries, dim)
            k: Количество ближайших соседей

        Returns:
            Кортеж (расстояния, индексы)

        Raises:
            AssertionError: Если индекс не построен/загружен
        """
        assert self._index is not None, "Index not built/loaded"
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def to_bytes(self) -> bytes:
        """Сериализует индекс в байты.

        Returns:
            Байты индекса

        Raises:
            AssertionError: Если индекс не построен/загружен
        """
        assert self._index is not None, "Index not built/loaded"
        vec = faiss.serialize_index(self._index)
        # vec is faiss.VectorUint8 -> конвертируем в байты через numpy для портабельности
        arr = np.array(vec, dtype="uint8")
        return bytes(arr)

    @classmethod
    def from_bytes(cls, data: bytes) -> VectorIndex:
        """Десериализует индекс из байтов.

        Args:
            data: Байты индекса

        Returns:
            Векторный индекс
        """
        # faiss ожидает numpy array буфер для десериализации
        arr = np.frombuffer(data, dtype="uint8")
        index = faiss.deserialize_index(arr)
        dim = index.d
        obj = cls(dim=dim)
        obj._index = index
        return obj


# ------------------------------ RAG оркестратор ---------------------- #


@dataclass
class RAGOrchestrator:
    """Оркестратор RAG пайплайна."""

    storage: StorageBackend
    embedder: EmbeddingsProvider
    index_metric: str = "ip"

    def build_and_save(
        self, docs: list[str], version: str, metadata: dict[str, Any]
    ) -> str:
        """Строит и сохраняет индекс для документов.

        Args:
            docs: Список документов
            version: Версия индекса
            metadata: Дополнительные метаданные

        Returns:
            Путь к сохраненному индексу
        """
        # Создаем эмбеддинги
        vectors = self.embedder.encode(docs)
        vdim = vectors.shape[1]

        # Строим индекс
        vindex = VectorIndex(dim=vdim, metric=self.index_metric)
        vindex.build(vectors)
        index_bytes = vindex.to_bytes()

        # Сохраняем индекс + метаданные
        meta = {
            "dim": vdim,
            "metric": self.index_metric,
            "docs": docs,
            **metadata,
        }
        return self.storage.save_index(version, index_bytes, meta)

    def load_index(self, version: str) -> VectorIndex:
        """Загружает индекс по версии.

        Args:
            version: Версия индекса

        Returns:
            Векторный индекс
        """
        idx_bytes, meta = self.storage.load_index(version)
        vindex = VectorIndex.from_bytes(idx_bytes)
        vindex.metadata = meta
        return vindex


# ------------------------------ Утилиты ----------------------------- #

DEFAULT_LOCAL_STORAGE = os.environ.get(
    "LOCAL_STORAGE_PATH", str(Path(__file__).resolve().parents[3] / "data" / "indexes")
)


def default_local_orchestrator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> RAGOrchestrator:
    """Создает оркестратор RAG с локальным хранилищем по умолчанию.

    Args:
        model_name: Название модели для эмбеддингов
        device: Устройство для вычислений (cpu/cuda)

    Returns:
        Оркестратор RAG
    """
    storage = LocalStorageBackend(base_path=Path(DEFAULT_LOCAL_STORAGE))
    embedder = EmbeddingsProvider(model_name=model_name, device=device)
    return RAGOrchestrator(storage=storage, embedder=embedder)
