# RAG core abstractions and local storage implementation
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple, Dict, Any, Optional

import faiss  # type: ignore
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------- Storage Layer ---------------------------- #

class StorageBackend(Protocol):
    def save_index(self, version: str, index_bytes: bytes, metadata: Dict[str, Any]) -> str: ...
    def load_index(self, version: str) -> Tuple[bytes, Dict[str, Any]]: ...
    def list_versions(self) -> list[str]: ...


@dataclass
class LocalStorageBackend:
    base_path: Path

    def __post_init__(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        d = self.base_path / f"v_{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_index(self, version: str, index_bytes: bytes, metadata: Dict[str, Any]) -> str:
        vdir = self._version_dir(version)
        index_path = vdir / "index.faiss"
        meta_path = vdir / "metadata.json"

        # atomic write
        tmp_index = vdir / "index.faiss.tmp"
        with open(tmp_index, "wb") as f:
            f.write(index_bytes)
        tmp_index.replace(index_path)

        tmp_meta = vdir / "metadata.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        tmp_meta.replace(meta_path)
        return str(vdir)

    def load_index(self, version: str) -> Tuple[bytes, Dict[str, Any]]:
        vdir = self._version_dir(version)
        index_path = vdir / "index.faiss"
        meta_path = vdir / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index or metadata not found for version {version}")
        with open(index_path, "rb") as f:
            idx_bytes = f.read()
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return idx_bytes, metadata

    def list_versions(self) -> list[str]:
        versions = []
        for p in self.base_path.glob("v_*"):
            if p.is_dir():
                versions.append(p.name.removeprefix("v_"))
        versions.sort()
        return versions


# ------------------------- Embeddings Provider ------------------------- #

@dataclass
class EmbeddingsProvider:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None  # e.g., "cpu" | "cuda"

    _model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        model = self._ensure_model()
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            # L2 normalize for cosine similarity with inner product index
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings.astype("float32")


# ------------------------------ VectorIndex --------------------------- #

@dataclass
class VectorIndex:
    dim: int
    metric: str = "ip"  # inner product (cosine when vectors are l2-normalized)
    _index: Optional[faiss.Index] = None

    def build(self, vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        if self.metric == "ip":
            index = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError("Unsupported metric: " + self.metric)
        index.add(vectors)
        self._index = index

    def search(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        assert self._index is not None, "Index not built/loaded"
        distances, indices = self._index.search(queries, k)
        return distances, indices

    def to_bytes(self) -> bytes:
        assert self._index is not None, "Index not built/loaded"
        vec = faiss.serialize_index(self._index)
        # vec is faiss.VectorUint8 -> convert to bytes via numpy for portability (Windows)
        arr = np.array(vec, dtype="uint8")
        return bytes(arr)

    @classmethod
    def from_bytes(cls, data: bytes) -> "VectorIndex":
        # faiss expects a numpy array buffer (uint8) for deserialization on some platforms (e.g., Windows)
        arr = np.frombuffer(data, dtype="uint8")
        index = faiss.deserialize_index(arr)
        dim = index.d
        obj = cls(dim=dim)
        obj._index = index
        return obj


# ------------------------------ RAG Orchestrator ---------------------- #

@dataclass
class RAGOrchestrator:
    storage: StorageBackend
    embedder: EmbeddingsProvider
    index_metric: str = "ip"

    def build_and_save(self, docs: list[str], version: str, metadata: Dict[str, Any]) -> str:
        # Create embeddings
        vectors = self.embedder.encode(docs)
        vdim = vectors.shape[1]

        # Build index
        vindex = VectorIndex(dim=vdim, metric=self.index_metric)
        vindex.build(vectors)
        index_bytes = vindex.to_bytes()

        # Save index + metadata (store doc ids if needed)
        meta = {
            "dim": vdim,
            "metric": self.index_metric,
            **metadata,
        }
        path = self.storage.save_index(version, index_bytes, meta)
        return path

    def load_index(self, version: str) -> VectorIndex:
        idx_bytes, meta = self.storage.load_index(version)
        vindex = VectorIndex.from_bytes(idx_bytes)
        # optional: validate meta["dim"] == vindex.dim
        return vindex


# ------------------------------ Utilities ----------------------------- #

DEFAULT_LOCAL_STORAGE = os.environ.get("LOCAL_STORAGE_PATH", str(Path(__file__).resolve().parents[3] / "data" / "indexes"))


def default_local_orchestrator(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                               device: Optional[str] = None) -> RAGOrchestrator:
    storage = LocalStorageBackend(base_path=Path(DEFAULT_LOCAL_STORAGE))
    embedder = EmbeddingsProvider(model_name=model_name, device=device)
    return RAGOrchestrator(storage=storage, embedder=embedder)
