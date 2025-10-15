import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import logging
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache
import PyPDF2
import docx
from io import BytesIO

from .models import (
    KnowledgeBase, DocumentChunk, RAGQuery, RAGConversation, 
    ConversationMessage, RAGSystemSettings
)

logger = logging.getLogger('apps.rag_assistant')

class RAGService:
    """Основной сервис RAG системы"""
    
    def __init__(self):
        self.settings = RAGSystemSettings.get_active_settings()
        if not self.settings:
            # Создать настройки по умолчанию
            self.settings = RAGSystemSettings.objects.create(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                is_active=True
            )
        
        self.embedding_model = None
        self._load_embedding_model()
        
        # Настройка OpenAI (если используется)
        if hasattr(settings, 'OPENAI_API_KEY'):
            openai.api_key = settings.OPENAI_API_KEY
    
    def _load_embedding_model(self):
        """Загрузка модели эмбеддингов"""
        try:
            if not self.embedding_model:
                logger.info(f"Загрузка модели эмбеддингов: {self.settings.embedding_model}")
                self.embedding_model = SentenceTransformer(self.settings.embedding_model)
                logger.info("Модель эмбеддингов успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            # Fallback к простейшей модели
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Загружена fallback модель эмбеддингов")
            except Exception as e2:
                logger.error(f"Критическая ошибка загрузки модели: {e2}")
                raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Генерация векторных представлений для текстов"""
        try:
            if not self.embedding_model:
                self._load_embedding_model()
            
            # Генерация эмбеддингов
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            # Возвращаем пустые эмбеддинги в случае ошибки
            return [[0.0] * self.settings.embedding_dimensions for _ in texts]
    
    def search_documents(self, query: str, category: Optional[str] = None, 
                        top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Семантический поиск документов"""
        try:
            # Генерация эмбеддинга для запроса
            query_embedding = self.generate_embeddings([query])
            
            # Фильтрация документов
            documents_query = KnowledgeBase.objects.filter(status='active')
            if category:
                documents_query = documents_query.filter(category=category)
            
            documents = documents_query.all()
            
            results = []
            
            for document in documents:
                # Поиск по фрагментам документа
                chunks = DocumentChunk.objects.filter(document=document)
                
                best_similarity = 0.0
                best_chunk = None
                
                for chunk in chunks:
                    if not chunk.embedding_vector:
                        continue
                    
                    # Вычисление косинусного сходства
                    similarity = self._calculate_similarity(query_embedding, chunk.embedding_vector)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_chunk = chunk
                
                # Если найден подходящий фрагмент
                if best_similarity >= min_similarity:
                    highlighted_content = self._highlight_relevant_content(
                        best_chunk.content, query
                    )
                    
                    results.append({
                        'document': document,
                        'similarity_score': best_similarity,
                        'matched_chunk': best_chunk,
                        'highlighted_content': highlighted_content
                    })
            
            # Сортировка по релевантности
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            return []
    
    def process_query(self, query: RAGQuery) -> Dict[str, Any]:
        """Обработка RAG запроса"""
        start_time = timezone.now()
        
        try:
            # Поиск релевантных документов
            search_results = self.search_documents(
                query.query_text,
                top_k=self.settings.search_top_k,
                min_similarity=self.settings.similarity_threshold
            )
            
            if not search_results:
                return {
                    'response': 'К сожалению, не удалось найти релевантные документы для вашего запроса. Попробуйте переформулировать вопрос.',
                    'confidence': 0.0,
                    'sources': [],
                    'processing_time': timezone.now() - start_time
                }
            
            # Подготовка контекста из найденных документов
            context_chunks = []
            sources = []
            
            for result in search_results:
                context_chunks.append(result['highlighted_content'])
                sources.append({
                    'document_id': result['document'].id,
                    'relevance_score': result['similarity_score'],
                    'chunk_id': result['matched_chunk'].id if result['matched_chunk'] else None
                })
            
            context = '\n\n'.join(context_chunks)
            
            # Генерация ответа
            response_text = self._generate_response(query.query_text, context, query.query_type)
            
            # Оценка уверенности ответа
            confidence = self._calculate_confidence(search_results, response_text)
            
            processing_time = timezone.now() - start_time
            
            return {
                'response': response_text,
                'confidence': confidence,
                'sources': sources,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса {query.id}: {e}")
            return {
                'response': f'Произошла ошибка при обработке запроса: {str(e)}',
                'confidence': 0.0,
                'sources': [],
                'processing_time': timezone.now() - start_time,
                'error': str(e)
            }
    
    def process_conversation_message(self, conversation: RAGConversation, 
                                   message_content: str) -> Dict[str, Any]:
        """Обработка сообщения в беседе"""
        start_time = timezone.now()
        
        try:
            # Получение контекста беседы
            previous_messages = ConversationMessage.objects.filter(
                conversation=conversation
            ).order_by('-created_at')[:10]  # Последние 10 сообщений
            
            # Формирование контекста беседы
            conversation_context = []
            for msg in reversed(previous_messages):
                role = "user" if msg.message_type == "user" else "assistant"
                conversation_context.append({
                    "role": role,
                    "content": msg.content
                })
            
            # Поиск релевантных документов для текущего сообщения
            search_results = self.search_documents(
                message_content,
                top_k=3,  # Меньше результатов для беседы
                min_similarity=self.settings.similarity_threshold
            )
            
            # Подготовка контекста из документов
            document_context = []
            source_documents = []
            
            if search_results:
                for result in search_results:
                    document_context.append(result['highlighted_content'])
                    source_documents.append(result['document'])
            
            # Генерация ответа с учетом контекста беседы
            response_text = self._generate_conversation_response(
                message_content, 
                conversation_context,
                document_context
            )
            
            processing_time = timezone.now() - start_time
            
            return {
                'response': response_text,
                'source_documents': source_documents,
                'response_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения в беседе {conversation.id}: {e}")
            return {
                'response': f'Произошла ошибка при обработке сообщения: {str(e)}',
                'source_documents': [],
                'response_time': timezone.now() - start_time
            }
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Вычисление косинусного сходства между векторами"""
        try:
            # Преобразование в numpy массивы
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Вычисление косинусного сходства
            similarity = cosine_similarity(vec1, vec2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Ошибка вычисления сходства: {e}")
            return 0.0
    
    def _highlight_relevant_content(self, content: str, query: str) -> str:
        """Выделение релевантного содержимого"""
        try:
            # Простое выделение ключевых слов из запроса
            query_words = re.findall(r'\w+', query.lower())
            
            # Поиск предложений, содержащих ключевые слова
            sentences = re.split(r'[.!?]+', content)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = re.findall(r'\w+', sentence.lower())
                
                # Подсчет совпадений
                matches = sum(1 for word in query_words if word in sentence_words)
                
                if matches > 0:
                    relevant_sentences.append({
                        'sentence': sentence.strip(),
                        'relevance': matches / len(query_words)
                    })
            
            # Сортировка по релевантности и выбор лучших
            relevant_sentences.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Возврат топ предложений
            if relevant_sentences:
                top_sentences = [s['sentence'] for s in relevant_sentences[:3]]
                return '. '.join(top_sentences)
            
            # Если не найдено релевантных предложений, вернуть начало документа
            return content[:500] + '...' if len(content) > 500 else content
            
        except Exception as e:
            logger.error(f"Ошибка выделения релевантного контента: {e}")
            return content[:500] + '...' if len(content) > 500 else content
    
    def _generate_response(self, query: str, context: str, query_type: str) -> str:
        """Генерация ответа на основе контекста"""
        try:
            # Проверка наличия API ключа OpenAI
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                return self._generate_openai_response(query, context, query_type)
            else:
                return self._generate_template_response(query, context, query_type)
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return self._generate_fallback_response(query, context)
    
    def _generate_openai_response(self, query: str, context: str, query_type: str) -> str:
        """Генерация ответа через OpenAI API"""
        try:
            # Определение системного промпта в зависимости от типа запроса
            system_prompts = {
                'question': "Вы - эксперт по гидравлическим системам. Отвечайте на вопросы на основе предоставленного контекста. Если информации недостаточно, сообщите об этом.",
                'analysis': "Вы - аналитик гидравлических систем. Проводите детальный анализ на основе технической документации.",
                'recommendation': "Вы - консультант по гидравлическим системам. Предоставляйте практические рекомендации на основе технических требований.",
                'search': "Вы - поисковый ассистент. Помогайте найти и структурировать релевантную информацию."
            }
            
            system_prompt = system_prompts.get(query_type, system_prompts['question'])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Контекст: {context}\n\nВопрос: {query}"}
                ],
                max_tokens=self.settings.max_response_tokens,
                temperature=self.settings.temperature
            )
            
            return response.choices.message.content.strip()
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа через OpenAI: {e}")
            return self._generate_template_response(query, context, query_type)
    
    def _generate_template_response(self, query: str, context: str, query_type: str) -> str:
        """Генерация шаблонного ответа"""
        templates = {
            'question': f"""На основе доступной технической документации:

{context}

В ответ на ваш вопрос "{query}" могу сообщить, что согласно нормативным документам и технической документации, необходимо руководствоваться указанными в контексте требованиями и процедурами.""",
            
            'analysis': f"""Анализ по запросу "{query}":

Основываясь на технической документации:
{context}

Рекомендую обратить внимание на ключевые аспекты, указанные в документации, и следовать установленным процедурам.""",
            
            'recommendation': f"""Рекомендации по запросу "{query}":

На основе анализа технической документации:
{context}

Рекомендую следовать указанным в документации процедурам и требованиям безопасности.""",
            
            'search': f"""По вашему запросу "{query}" найдена следующая информация:

{context}

Для получения дополнительной информации рекомендую изучить полные версии соответствующих документов."""
        }
        
        return templates.get(query_type, templates['question'])
    
    def _generate_conversation_response(self, message: str, conversation_context: List[Dict], 
                                      document_context: List[str]) -> str:
        """Генерация ответа в контексте беседы"""
        try:
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                # Формирование сообщений для OpenAI
                messages = [
                    {
                        "role": "system", 
                        "content": "Вы - ассистент по гидравлическим системам. Отвечайте на основе технической документации и контекста беседы."
                    }
                ]
                
                # Добавление контекста беседы
                messages.extend(conversation_context)
                
                # Добавление контекста из документов
                if document_context:
                    context_text = "\n\n".join(document_context)
                    messages.append({
                        "role": "system",
                        "content": f"Дополнительный контекст из документации: {context_text}"
                    })
                
                # Добавление текущего сообщения
                messages.append({"role": "user", "content": message})
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=self.settings.max_response_tokens,
                    temperature=self.settings.temperature
                )
                
                return response.choices.message.content.strip()
            else:
                # Простой шаблонный ответ
                if document_context:
                    context_text = "\n\n".join(document_context[:2])  # Первые 2 контекста
                    return f"""На основе найденной документации:

{context_text}

Рекомендую ознакомиться с полной версией соответствующих документов для получения детальной информации по вашему запросу "{message}"."""
                else:
                    return f"""По вашему запросу "{message}" в данный момент не найдено релевантных документов в базе знаний. 

Рекомендую:
1. Переформулировать запрос
2. Использовать более общие термины
3. Обратиться к администратору для добавления соответствующей документации"""
                    
        except Exception as e:
            logger.error(f"Ошибка генерации ответа в беседе: {e}")
            return f"Произошла ошибка при генерации ответа. Попробуйте переформулировать вопрос."
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Резервный ответ при ошибках"""
        return f"""На основе найденной документации:

{context[:1000]}...

Для получения более детальной информации по запросу "{query}" рекомендую ознакомиться с полными версиями соответствующих документов."""
    
    def _calculate_confidence(self, search_results: List[Dict], response_text: str) -> float:
        """Вычисление уверенности в ответе"""
        try:
            if not search_results:
                return 0.0
            
            # Факторы уверенности
            avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
            result_count_factor = min(len(search_results) / 5.0, 1.0)  # Нормализация к 5 результатам
            response_length_factor = min(len(response_text) / 500.0, 1.0)  # Нормализация к 500 символам
            
            # Итоговая уверенность
            confidence = (avg_similarity * 0.6 + result_count_factor * 0.2 + response_length_factor * 0.2)
            
            return min(max(confidence, 0.0), 1.0)  # Ограничение 0-1
            
        except Exception as e:
            logger.error(f"Ошибка вычисления уверенности: {e}")
            return 0.5  # Средняя уверенность по умолчанию


class DocumentProcessor:
    """Сервис для обработки документов"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.settings = RAGSystemSettings.get_active_settings()
        if not self.settings:
            self.settings = RAGSystemSettings.objects.create(is_active=True)
    
    def process_uploaded_file(self, file, title: Optional[str] = None, 
                            category: str = 'manual', description: str = '',
                            document_number: str = '', uploaded_by=None) -> KnowledgeBase:
        """Обработка загруженного файла"""
        try:
            # Извлечение текста из файла
            content = self._extract_text_from_file(file)
            
            # Определение названия документа
            if not title:
                title = file.name
            
            # Создание документа
            document = KnowledgeBase.objects.create(
                title=title,
                category=category,
                description=description,
                content=content,
                document_number=document_number,
                uploaded_by=uploaded_by,
                status='processing'
            )
            
            # Обработка документа (создание фрагментов и эмбеддингов)
            self._process_document_content(document)
            
            # Отметить документ как активный
            document.status = 'active'
            document.save()
            
            logger.info(f"Документ {document.id} успешно обработан")
            return document
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file.name}: {e}")
            if 'document' in locals():
                document.status = 'error'
                document.processing_notes = str(e)
                document.save()
            raise
    
    def _extract_text_from_file(self, file) -> str:
        """Извлечение текста из различных типов файлов"""
        content_type = file.content_type
        
        try:
            if content_type == 'text/plain':
                return file.read().decode('utf-8', errors='ignore')
            
            elif content_type == 'application/pdf':
                return self._extract_from_pdf(file)
            
            elif content_type in ['application/msword', 
                                'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return self._extract_from_docx(file)
            
            else:
                # Попытка прочитать как текст
                return file.read().decode('utf-8', errors='ignore')
                
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из файла: {e}")
            raise ValueError(f"Не удалось извлечь текст из файла: {e}")
    
    def _extract_from_pdf(self, file) -> str:
        """Извлечение текста из PDF файла"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {e}")
            raise
    
    def _extract_from_docx(self, file) -> str:
        """Извлечение текста из DOCX файла"""
        try:
            doc = docx.Document(BytesIO(file.read()))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из DOCX: {e}")
            raise
    
    def _process_document_content(self, document: KnowledgeBase):
        """Обработка содержимого документа (создание фрагментов и эмбеддингов)"""
        try:
            # Разделение на фрагменты
            chunks = self._split_text_into_chunks(document.content)
            
            # Создание фрагментов в БД
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = DocumentChunk.objects.create(
                    document=document,
                    content=chunk_text,
                    chunk_index=i,
                    start_position=0,  # Можно улучшить для точного позиционирования
                    end_position=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                )
                document_chunks.append(chunk)
            
            # Генерация эмбеддингов для всех фрагментов
            chunk_texts = [chunk.content for chunk in document_chunks]
            embeddings = self.rag_service.generate_embeddings(chunk_texts)
            
            # Сохранение эмбеддингов
            for chunk, embedding in zip(document_chunks, embeddings):
                chunk.embedding_vector = embedding
                chunk.save()
            
            # Генерация эмбеддинга для всего документа
            document_embedding = self.rag_service.generate_embeddings([document.content])
            document.embedding_vector = document_embedding
            document.embedding_model = self.settings.embedding_model
            
            # Генерация краткого содержания и ключевых слов
            document.summary = self._generate_summary(document.content)
            document.keywords = self._extract_keywords(document.content)
            
            document.save()
            
            logger.info(f"Создано {len(document_chunks)} фрагментов для документа {document.id}")
            
        except Exception as e:
            logger.error(f"Ошибка обработки содержимого документа {document.id}: {e}")
            raise
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Разделение текста на фрагменты"""
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap
        
        # Простое разделение по предложениям с учетом размера
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Проверка, помещается ли предложение в текущий фрагмент
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохранить текущий фрагмент
                if current_chunk:
                    chunks.append(current_chunk + ".")
                
                # Начать новый фрагмент
                current_chunk = sentence
        
        # Добавить последний фрагмент
        if current_chunk:
            chunks.append(current_chunk + ".")
        
        return chunks
    
    def _generate_summary(self, content: str) -> str:
        """Генерация краткого содержания документа"""
        try:
            # Простое извлечение первых предложений
            sentences = re.split(r'[.!?]+', content)
            summary_sentences = []
            
            for sentence in sentences[:5]:  # Первые 5 предложений
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:  # Игнорируем очень короткие предложения
                    summary_sentences.append(sentence)
            
            summary = '. '.join(summary_sentences)
            return summary[:500] + '...' if len(summary) > 500 else summary
            
        except Exception as e:
            logger.error(f"Ошибка генерации summary: {e}")
            return content[:200] + '...' if len(content) > 200 else content
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Извлечение ключевых слов из документа"""
        try:
            # Простое извлечение наиболее частых слов
            words = re.findall(r'\b[а-яё]{4,}\b', content.lower())
            
            # Подсчет частотности
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Сортировка по частотности и выбор топ-20
            sorted_words = sorted(word_freq.items(), key=lambda x: x, reverse=True)
            keywords = [word for word, freq in sorted_words[:20] if freq > 1]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых слов: {e}")
            return []
    
    def reprocess_document(self, document: KnowledgeBase):
        """Переобработка существующего документа"""
        try:
            # Удаление существующих фрагментов
            DocumentChunk.objects.filter(document=document).delete()
            
            # Обновление статуса
            document.status = 'processing'
            document.save()
            
            # Повторная обработка
            self._process_document_content(document)
            
            # Отметить как активный
            document.status = 'active'
            document.save()
            
            logger.info(f"Документ {document.id} успешно переобработан")
            
        except Exception as e:
            document.status = 'error'
            document.processing_notes = str(e)
            document.save()
            logger.error(f"Ошибка переобработки документа {document.id}: {e}")
            raise
