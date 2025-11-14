"""
Embeddings для RAG Knowledge Base.
Использует multilingual-e5-large для векторизации документов.
"""
import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """
    Сервис для генерации embeddings.
    
    Модель: intfloat/multilingual-e5-large
    - Размер: 1.5GB
    - Векторы: 1024 dimensions
    - Языки: 100+ (включая русский)
    - Device: CPU (не требует GPU!)
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cpu",  # Используем CPU, GPU для DeepSeek
        cache_dir: str = "/app/models/embeddings"
    ):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embeddings model: {model_name} on {device}")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )
        
        # Проверка размерности
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embeddings dimension: {self.embedding_dim}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Векторизация поискового запроса.
        
        Args:
            text: Текст запроса
            
        Returns:
            np.ndarray: Вектор (1024,)
        """
        # E5 требует префикс "query: " для поисковых запросов
        query_text = f"query: {text}"
        
        with torch.no_grad():
            embedding = self.model.encode(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        
        return embedding
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Векторизация документов для индексации.
        
        Args:
            texts: Список текстов документов
            batch_size: Размер батча
            show_progress: Показывать прогресс
            
        Returns:
            np.ndarray: Матрица векторов (n_docs, 1024)
        """
        # E5 требует префикс "passage: " для документов
        passages = [f"passage: {text}" for text in texts]
        
        logger.info(f"Embedding {len(texts)} documents...")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                passages,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
        
        logger.info(f"Embedded {len(texts)} documents → {embeddings.shape}")
        return embeddings
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Вычисление cosine similarity между запросом и документами.
        
        Args:
            query_embedding: Вектор запроса (1024,)
            doc_embeddings: Матрица векторов документов (n, 1024)
            
        Returns:
            np.ndarray: Scores (n,)
        """
        # Normalized vectors → cosine = dot product
        scores = np.dot(doc_embeddings, query_embedding)
        return scores


# Global singleton
_embeddings_service: Optional[EmbeddingsService] = None


def get_embeddings_service() -> EmbeddingsService:
    """
    Get global embeddings service instance.
    
    Returns:
        EmbeddingsService: Initialized service
    """
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service
