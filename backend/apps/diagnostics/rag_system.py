import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HydraulicKnowledgeBase:
    """RAG система для технических знаний по гидравлике"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words=None, ngram_range=(1, 2)
        )
        self.knowledge_vectors = None
        self.knowledge_texts: List[str] = []
        self.knowledge_metadata: List[Dict[str, Any]] = []
        self.db_path = os.path.join(settings.BASE_DIR, "knowledge_base.db")
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Инициализация базы знаний"""
        try:
            # Создание базы данных если не существует
            self._create_database()

            # Загрузка базовых знаний
            self._load_base_knowledge()

            # Построение векторов
            if self.knowledge_texts:
                self._build_vectors()
                logger.info(
                    f"Загружено {len(self.knowledge_texts)} документов в базу знаний"
                )

        except Exception as e:
            logger.error(f"Ошибка инициализации базы знаний: {e}")

    def _create_database(self):
        """Создание базы данных для хранения знаний"""
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

    def _load_base_knowledge(self):
        """Загрузка базовых знаний о гидравлических системах"""
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
            {
                "title": "Температурная диагностика гидросистем",
                "content": (
                    "Нормальная рабочая температура гидравлического масла: 40-60°C. "
                    "Температура выше 70°C указывает на перегрузку системы, засорение радиатора или износ уплотнений. "
                    "Температура ниже 20°C может вызвать загустение масла и кавитацию насоса. Резкие перепады температуры "
                    "свидетельствуют о проблемах терморегуляции."
                ),
                "category": "diagnostics",
                "tags": "температура, масло, перегрузка, радиатор, уплотнения",
                "source": "technical_manual",
            },
            {
                "title": "Анализ расхода жидкости",
                "content": (
                    "Номинальный расход для промышленных систем: 50-100 л/мин. "
                    "Снижение расхода более чем на 15% указывает на износ насоса, засорение фильтров или утечки. "
                    "Увеличение расхода может быть связано с неплотностями в системе. Пульсации расхода указывают на "
                    "воздух в системе или неисправность насоса."
                ),
                "category": "diagnostics",
                "tags": "расход, насос, фильтр, утечки, пульсации",
                "source": "technical_manual",
            },
            {
                "title": "Вибродиагностика гидроагрегатов",
                "content": (
                    "Нормальный уровень вибрации для гидронасосов: до 7,1 мм/с (СКЗ). "
                    "Вибрация 7,1-18 мм/с - удовлетворительное состояние. Свыше 18 мм/с - недопустимо. "
                    "Частоты вибрации: 1х оборотная - дисбаланс, 2х оборотная - несоосность, "
                    "высокочастотная - износ подшипников, кавитация."
                ),
                "category": "diagnostics",
                "tags": "вибрация, насос, подшипники, дисбаланс, кавитация",
                "source": "vibration_manual",
            },
            {
                "title": "Типовые неисправности промышленных гидросистем",
                "content": (
                    "1. Износ плунжеров насоса - снижение давления и расхода, металлическая стружка в масле. "
                    "2. Засорение фильтров - рост давления, падение расхода, перегрев. "
                    "3. Износ уплотнений - внешние утечки, падение давления, загрязнение масла. "
                    "4. Кавитация насоса - характерный шум, вибрация, эрозия металла. "
                    "5. Загрязнение масла - ускоренный износ, перегрев, снижение эффективности."
                ),
                "category": "failures",
                "tags": "неисправности, насос, фильтры, уплотнения, кавитация, масло",
                "source": "failure_analysis",
            },
            {
                "title": "Рекомендации по техническому обслуживанию",
                "content": (
                    "Регламентные работы каждые 500 часов: замена масла, проверка фильтров, "
                    "контроль давления. Каждые 1000 часов: замена фильтров, проверка уплотнений, "
                    "вибродиагностика. Каждые 2000 часов: ревизия насоса, проверка аккумуляторов, "
                    "промывка системы. Аварийное обслуживание при превышении пороговых значений."
                ),
                "category": "maintenance",
                "tags": "обслуживание, масло, фильтры, насос, регламент",
                "source": "maintenance_manual",
            },
        ]

        # Сохранение в базу данных
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

        # Загрузка в память
        self.knowledge_texts = [doc["content"] for doc in base_knowledge]
        self.knowledge_metadata = base_knowledge

    def _build_vectors(self):
        """Построение векторных представлений документов"""
        try:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_texts)
            logger.info("Векторы знаний построены успешно")
        except Exception as e:
            logger.error(f"Ошибка построения векторов: {e}")

    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск релевантных знаний по запросу"""
        try:
            if self.knowledge_vectors is None or not query.strip():
                return []

            # Векторизация запроса
            query_vector = self.vectorizer.transform([query])

            # Вычисление косинусного сходства
            similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]

            # Получение топ-K наиболее релевантных документов
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results: List[Dict[str, Any]] = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Порог релевантности
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

    def get_diagnostic_recommendations(
        self, symptoms: List[str], system_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение рекомендаций по диагностике на основе симптомов"""
        try:
            # Формирование поискового запроса
            query = " ".join(symptoms)
            if system_type:
                query += f" {system_type}"

            # Поиск релевантных документов
            relevant_docs = self.search_knowledge(query, top_k=3)

            recommendations: List[Dict[str, Any]] = []
            for doc in relevant_docs:
                # Извлечение рекомендаций из содержимого
                content = doc["content"]

                if "диагностика" in content.lower() or "проверка" in content.lower():
                    recommendations.append(
                        {
                            "title": f"Рекомендация на основе: {doc['title']}",
                            "description": self._extract_recommendations(content),
                            "source": doc["source"],
                            "relevance": doc["relevance_score"],
                            "category": doc["category"],
                        }
                    )

            return recommendations

        except Exception as e:
            logger.error(f"Ошибка получения рекомендаций: {e}")
            return []

    def add_diagnostic_case(
        self,
        problem: str,
        symptoms: List[str],
        solution: str,
        system_type: Optional[str] = None,
        success_rate: float = 1.0,
    ) -> None:
        """Добавление нового диагностического случая в базу знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO diagnostic_cases
                (problem_description, symptoms, solution, system_type, success_rate)
                VALUES (?, ?, ?, ?, ?)
            """,
                (problem, json.dumps(symptoms), solution, system_type, success_rate),
            )

            conn.commit()
            conn.close()

            # Добавление в текущую сессию
            new_doc = {
                "title": f"Диагностический случай: {problem}",
                "content": f"{problem} {' '.join(symptoms)} {solution}",
                "category": "diagnostic_case",
                "tags": ", ".join(symptoms),
                "source": "user_case",
            }

            self.knowledge_texts.append(new_doc["content"])
            self.knowledge_metadata.append(new_doc)

            # Перестроение векторов
            self._build_vectors()

            logger.info(f"Добавлен новый диагностический случай: {problem}")

        except Exception as e:
            logger.error(f"Ошибка добавления случая: {e}")

    def get_similar_cases(
        self, current_symptoms: List[str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Поиск похожих диагностических случаев"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM diagnostic_cases ORDER BY created_at DESC LIMIT 100"
            )
            cases = cursor.fetchall()
            conn.close()

            similar_cases: List[Dict[str, Any]] = []
            current_symptoms_set = set(current_symptoms)

            for case in cases:
                case_symptoms = json.loads(case[2]) if case[2] else []
                case_symptoms_set = set(case_symptoms)

                # Вычисление сходства по Жаккару
                intersection = len(current_symptoms_set.intersection(case_symptoms_set))
                union = len(current_symptoms_set.union(case_symptoms_set))
                similarity = intersection / union if union > 0 else 0

                if similarity > 0.3:  # Порог сходства
                    similar_cases.append(
                        {
                            "problem": case[1],
                            "symptoms": case_symptoms,
                            "solution": case[3],
                            "system_type": case[4],
                            "similarity": similarity,
                            "success_rate": case[5],
                        }
                    )

            # Сортировка по сходству
            similar_cases.sort(key=lambda x: x["similarity"], reverse=True)

            return similar_cases[:top_k]

        except Exception as e:
            logger.error(f"Ошибка поиска похожих случаев: {e}")
            return []

    def generate_contextual_answer(
        self, question: str, context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Генерация контекстного ответа на основе базы знаний"""
        try:
            # Поиск релевантной информации
            relevant_docs = self.search_knowledge(question, top_k=5)

            if not relevant_docs:
                return {
                    "answer": "К сожалению, по данному вопросу информация в базе знаний не найдена.",
                    "confidence": 0.0,
                    "sources": [],
                }

            # Формирование ответа на основе найденной информации
            answer_parts: List[str] = []
            sources: List[Dict[str, Any]] = []
            total_relevance = 0.0

            for doc in relevant_docs:
                # Извлечение наиболее релевантной части документа
                relevant_part = self._extract_relevant_part(doc["content"], question)
                if relevant_part:
                    answer_parts.append(relevant_part)
                    sources.append(
                        {
                            "title": doc["title"],
                            "source": doc["source"],
                            "relevance": doc["relevance_score"],
                        }
                    )
                    total_relevance += float(doc["relevance_score"])  # типобезопасно

            # Объединение частей ответа
            if answer_parts:
                answer = " ".join(answer_parts[:3])  # Максимум 3 части
                confidence = min(total_relevance / len(answer_parts), 1.0)
            else:
                answer = "Информация найдена, но требует дополнительного анализа."
                confidence = 0.3

            # Нормализация контекста
            context = context_data or {}

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "context": context,
            }

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return {
                "answer": "Произошла ошибка при обработке запроса.",
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
            }

    def _extract_recommendations(self, content: str) -> str:
        """Извлечение рекомендаций из содержимого документа"""
        # Простое извлечение предложений с ключевыми словами
        sentences = content.split(". ")
        recommendations: List[str] = []

        keywords = [
            "рекомендуется",
            "необходимо",
            "следует",
            "проверить",
            "заменить",
            "контролировать",
        ]

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                recommendations.append(sentence.strip())

        return (
            ". ".join(recommendations[:2]) if recommendations else content[:200] + "..."
        )

    def _extract_relevant_part(self, content: str, question: str) -> str:
        """Извлечение наиболее релевантной части документа для ответа"""
        sentences = content.split(". ")
        question_words = set(question.lower().split())

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))

            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        return best_sentence.strip() if best_sentence else content[:150] + "..."

    def update_knowledge_base(self, new_documents: List[Dict[str, Any]]) -> None:
        """Обновление базы знаний новыми документами"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for doc in new_documents:
                cursor.execute(
                    """
                    INSERT INTO knowledge_documents
                    (title, content, category, tags, source)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        doc["title"],
                        doc["content"],
                        doc.get("category", "general"),
                        doc.get("tags", ""),
                        doc.get("source", "user"),
                    ),
                )

                # Добавление в текущую сессию
                self.knowledge_texts.append(doc["content"])
                self.knowledge_metadata.append(doc)

            conn.commit()
            conn.close()

            # Перестроение векторов
            self._build_vectors()

            logger.info(f"База знаний обновлена {len(new_documents)} документами")

        except Exception as e:
            logger.error(f"Ошибка обновления базы знаний: {e}")

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Получение статистики базы знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM knowledge_documents")
            doc_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM diagnostic_cases")
            case_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT category, COUNT(*) FROM knowledge_documents GROUP BY category"
            )
            categories = dict(cursor.fetchall())

            conn.close()

            return {
                "total_documents": doc_count,
                "diagnostic_cases": case_count,
                "categories": categories,
                "vector_dimensions": (
                    self.vectorizer.max_features
                    if hasattr(self.vectorizer, "max_features")
                    else 0
                ),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}


# Глобальный экземпляр RAG системы
rag_system = HydraulicKnowledgeBase()
