"""Модуль проекта с автогенерированным докстрингом."""

import logging
import os
import sqlite3
from typing import Any

from django.conf import settings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HydraulicKnowledgeBase:
    """RAG система для технических знаний по гидравлике."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words=None, ngram_range=(1, 2)
        )
        self.knowledge_vectors = None
        self.knowledge_texts: list[str] = []
        self.knowledge_metadata: list[dict[str, Any]] = []
        self.db_path = os.path.join(settings.BASE_DIR, "knowledge_base.db")
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> None:
        """Инициализация базы знаний."""
        try:
            self._create_database()
            self._load_base_knowledge()
            if self.knowledge_texts:
                self._build_vectors()
                logger.info(
                    f"Загружено {len(self.knowledge_texts)} документов в базу знаний"
                )
        except Exception as e:
            logger.error(f"Ошибка инициализации базы знаний: {e}")

    def _create_database(self) -> None:
        """Создание базы данных для хранения знаний."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostic_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_description TEXT NOT NULL,
                symptoms TEXT,
                solution TEXT NOT NULL,
                system_type TEXT,
                success_rate REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_base_knowledge(self) -> None:
        """Загрузка базовых знаний о гидравлических системах."""
        base_knowledge = [
            {
                "title": "ГОСТ 17752-81 Гидропривод объемный. Термины и определения",
                "content": (
                    "Гидропривод объемный - совокупность устройств, предназначенных для приведения в движение "
                    "машин и механизмов посредством рабочей жидкости под давлением. Основные параметры: рабочее давление "
                    "от 6,3 до 32 МПа, температура рабочей жидкости от -30 до +80°C. Критические показатели: "
                    "падение давления более 10%, повышение температуры выше 80°C, "
                    "появление металлической стружки в масле."
                ),
                "category": "standards",
                "tags": "ГОСТ, гидропривод, давление, температура",
                "source": "ГОСТ 17752-81",
            },
            {
                "title": "Диагностика по давлению в гидросистеме",
                "content": (
                    "Нормальное рабочее давление составляет 150-250 бар для промышленных систем. "
                    "Давление ниже 100 бар может указывать на износ насоса или утечки. Давление выше 300 бар - "
                    "на засорение фильтров или неисправность предохранительного клапана. Колебания давления более ±5% "
                    "от номинального указывают на нестабильность системы."
                ),
                "category": "diagnostics",
                "tags": "давление, насос, утечки, клапан, диагностика",
                "source": "technical_manual",
            },
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for doc in base_knowledge:
            cursor.execute(
                """
                INSERT OR REPLACE INTO knowledge_documents
                (title, content, category, tags, source)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    doc["title"],
                    doc["content"],
                    doc["category"],
                    doc["tags"],
                    doc["source"],
                ),
            )

        conn.commit()
        conn.close()

        self.knowledge_texts = [doc["content"] for doc in base_knowledge]
        self.knowledge_metadata = base_knowledge

    def _build_vectors(self) -> None:
        """Построение векторных представлений документов."""
        try:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_texts)
            logger.info("Векторы знаний построены успешно")
        except Exception as e:
            logger.error(f"Ошибка построения векторов: {e}")

    def search_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск релевантных знаний по запросу."""
        try:
            if self.knowledge_vectors is None or not query.strip():
                return []
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results: list[dict[str, Any]] = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append(
                        {
                            "title": self.knowledge_metadata[idx]["title"],
                            "content": self.knowledge_metadata[idx]["content"],
                            "category": self.knowledge_metadata[idx]["category"],
                            "relevance_score": float(similarities[idx]),
                            "source": self.knowledge_metadata[idx]["source"],
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска знаний: {e}")
            return []


# Глобальный экземпляр RAG системы
rag_system = HydraulicKnowledgeBase()
