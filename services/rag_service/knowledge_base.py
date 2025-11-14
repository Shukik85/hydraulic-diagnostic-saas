"""
Knowledge Base с FAISS для RAG.
Управление базой знаний по гидравлике и maintenance.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
import faiss

from embeddings import get_embeddings_service

logger = logging.getLogger(__name__)


class Document:
    """Документ в базе знаний."""
    
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None
    ):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def to_dict(self) -> Dict:
        """Сериализация в dict (без embedding)."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """Десериализация из dict."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )


class KnowledgeBase:
    """
    База знаний с FAISS индексом.
    
    Хранит:
    - Векторный индекс (FAISS)
    - Метаданные документов (JSON)
    - Текстовое содержимое
    """
    
    def __init__(
        self,
        index_path: str = "/app/vector_store",
        index_name: str = "hydraulics"
    ):
        self.index_path = Path(index_path)
        self.index_name = index_name
        
        # Файлы индекса
        self.faiss_index_file = self.index_path / f"{index_name}.index"
        self.documents_file = self.index_path / f"{index_name}_docs.json"
        
        # Создаём директорию если нужно
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Embeddings service
        self.embedder = get_embeddings_service()
        self.embedding_dim = self.embedder.embedding_dim
        
        # FAISS index
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner Product (cosine for normalized)
        self.documents: List[Document] = []
        
        # Загружаем существующий индекс если есть
        if self.faiss_index_file.exists():
            self.load()
        else:
            logger.info(f"Creating new FAISS index: {index_name}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 32
    ):
        """
        Добавить документы в индекс.
        
        Args:
            documents: Список документов
            batch_size: Размер батча для embeddings
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to knowledge base...")
        
        # Генерируем embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed_documents(
            texts,
            batch_size=batch_size
        )
        
        # Добавляем в FAISS
        self.index.add(embeddings.astype(np.float32))
        
        # Сохраняем документы с embeddings
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
        
        logger.info(f"Knowledge base now contains {len(self.documents)} documents")
        
        # Автосохранение
        self.save()
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Поиск релевантных документов.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            threshold: Минимальный score (0-1)
            
        Returns:
            List[Tuple[Document, float]]: [(doc, score), ...]
        """
        if self.index.ntotal == 0:
            logger.warning("Knowledge base is empty")
            return []
        
        # Векторизация запроса
        query_embedding = self.embedder.embed_query(query)
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Поиск в FAISS
        scores, indices = self.index.search(query_vector, top_k)
        
        # Фильтрация и сборка результатов
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS возвращает -1 если не нашёл
                continue
            
            if score < threshold:
                continue
            
            doc = self.documents[idx]
            results.append((doc, float(score)))
        
        logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
        return results
    
    def save(self):
        """Сохранить индекс и документы на диск."""
        logger.info(f"Saving knowledge base to {self.index_path}")
        
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, str(self.faiss_index_file))
        
        # Сохраняем документы (без embeddings для экономии места)
        docs_data = [doc.to_dict() for doc in self.documents]
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved {len(self.documents)} documents")
    
    def load(self):
        """Загрузить индекс и документы с диска."""
        logger.info(f"Loading knowledge base from {self.index_path}")
        
        # Загружаем FAISS индекс
        self.index = faiss.read_index(str(self.faiss_index_file))
        
        # Загружаем документы
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        self.documents = [Document.from_dict(d) for d in docs_data]
        
        logger.info(f"✅ Loaded {len(self.documents)} documents")
    
    def clear(self):
        """Очистить весь индекс."""
        logger.warning("Clearing knowledge base...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        
        # Удаляем файлы
        if self.faiss_index_file.exists():
            self.faiss_index_file.unlink()
        if self.documents_file.exists():
            self.documents_file.unlink()
    
    def stats(self) -> Dict:
        """Статистика базы знаний."""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "index_path": str(self.index_path),
            "index_name": self.index_name
        }


# Global singleton
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(
    index_path: str = None,
    index_name: str = "hydraulics"
) -> KnowledgeBase:
    """
    Get global knowledge base instance.
    
    Args:
        index_path: Path to index directory
        index_name: Name of the index
        
    Returns:
        KnowledgeBase: Initialized knowledge base
    """
    global _knowledge_base
    if _knowledge_base is None:
        kb_path = index_path or os.getenv("VECTOR_STORE_PATH", "/app/vector_store")
        _knowledge_base = KnowledgeBase(index_path=kb_path, index_name=index_name)
    return _knowledge_base
