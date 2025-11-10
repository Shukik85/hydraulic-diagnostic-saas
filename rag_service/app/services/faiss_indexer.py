"""FAISS Indexer: загрузка и обновление индекса документов для RAG Service."""
import os
import structlog
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
import pickle
logger = structlog.get_logger(__name__)
class FAISSIndexer:
    def __init__(self, index_path: str, model_name: str):
        self.index_path = index_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docinfo = []  # Список метаданных docs (для поиска/ответа)
    def load_index(self) -> int:
        """Загрузка существующего индекса."""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            info_path = self.index_path + '.info'
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    self.docinfo = pickle.load(f)
            logger.info("FAISS index loaded", path=self.index_path, doccount=len(self.docinfo))
            return len(self.docinfo)
        else:
            logger.warning("No FAISS index found", path=self.index_path)
            self.index = None
            self.docinfo = []
            return 0
    def save_index(self):
        """Сохранить индекс и метаданные."""
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.index_path + '.info', 'wb') as f:
            pickle.dump(self.docinfo, f)
        logger.info("FAISS index saved", path=self.index_path, doccount=len(self.docinfo))
    def build_index_from_docs(self, docs: List[Dict[str, Any]], batch_size: int = 16):
        """Импорт документов: docs — список {text, meta}."""
        texts = [doc['text'] for doc in docs]
        metas = [doc.get('meta', {}) for doc in docs]
        # Генерация эмбеддингов
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        dim = embeddings.shape[1]
        # Создание FAISS индекса (Flat/IP-вектор)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.docinfo = metas
        self.save_index()
        logger.info("FAISS index built", doc_count=len(texts), dim=dim)
    def add_docs(self, docs: List[Dict[str, Any]], batch_size: int = 16):
        """Добавить новые документы в существующий индекс."""
        if self.index is None:
            self.build_index_from_docs(docs, batch_size)
            return
        texts = [doc['text'] for doc in docs]
        metas = [doc.get('meta', {}) for doc in docs]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        self.index.add(embeddings)
        self.docinfo.extend(metas)
        self.save_index()
        logger.info("Added new docs to FAISS index", added=len(texts), total=len(self.docinfo))
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск по запросу, возвращает top_k docs + score."""
        if self.index is None:
            logger.warning("Empty FAISS index, cannot search")
            return []
        query_emb = self.model.encode([query])[0]
        D, I = self.index.search([query_emb], top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            info = self.docinfo[idx] if idx < len(self.docinfo) else {}
            results.append({'score': float(score), 'meta': info})
        logger.info("FAISS search", query=query, results=len(results))
        return results
