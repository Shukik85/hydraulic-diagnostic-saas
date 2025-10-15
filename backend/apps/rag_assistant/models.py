from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
import json

User = get_user_model()

class KnowledgeBase(models.Model):
    """База знаний документов"""
    CATEGORY_CHOICES = [
        ('gost', 'ГОСТ'),
        ('manual', 'Руководство'),
        ('specification', 'Спецификация'),
        ('procedure', 'Процедура'),
        ('troubleshooting', 'Устранение неисправностей'),
        ('maintenance', 'Техническое обслуживание'),
        ('safety', 'Безопасность'),
        ('regulation', 'Нормативы'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Активен'),
        ('inactive', 'Неактивен'),
        ('processing', 'Обработка'),
        ('error', 'Ошибка'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500, verbose_name='Название документа')
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, verbose_name='Категория')
    description = models.TextField(blank=True, verbose_name='Описание')
    
    # Содержимое документа
    content = models.TextField(verbose_name='Содержимое документа')
    summary = models.TextField(blank=True, verbose_name='Краткое содержание')
    keywords = models.JSONField(default=list, blank=True, verbose_name='Ключевые слова')
    
    # Метаданные документа
    document_type = models.CharField(max_length=100, blank=True, verbose_name='Тип документа')
    document_number = models.CharField(max_length=100, blank=True, verbose_name='Номер документа')
    version = models.CharField(max_length=50, blank=True, verbose_name='Версия')
    publication_date = models.DateField(null=True, blank=True, verbose_name='Дата публикации')
    
    # Векторные представления для поиска
    embedding_vector = models.JSONField(default=list, blank=True, verbose_name='Векторное представление')
    embedding_model = models.CharField(max_length=100, blank=True, verbose_name='Модель эмбеддингов')
    
    # Статус и обработка
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name='Статус')
    processing_notes = models.TextField(blank=True, verbose_name='Заметки обработки')
    
    # Статистика использования
    search_count = models.PositiveIntegerField(default=0, verbose_name='Количество поисков')
    relevance_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name='Оценка релевантности'
    )
    
    # Автор и даты
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name='Загружен пользователем')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создан')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлен')
    
    class Meta:
        verbose_name = 'Документ базы знаний'
        verbose_name_plural = 'База знаний'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['category', 'status']),
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['document_number']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.get_category_display()})"
    
    def increment_search_count(self):
        """Увеличить счетчик поисков"""
        self.search_count += 1
        self.save(update_fields=['search_count'])

class DocumentChunk(models.Model):
    """Фрагменты документов для векторного поиска"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE, related_name='chunks', verbose_name='Документ')
    
    # Содержимое фрагмента
    content = models.TextField(verbose_name='Содержимое фрагмента')
    chunk_index = models.PositiveIntegerField(verbose_name='Индекс фрагмента')
    start_position = models.PositiveIntegerField(verbose_name='Начальная позиция')
    end_position = models.PositiveIntegerField(verbose_name='Конечная позиция')
    
    # Векторное представление
    embedding_vector = models.JSONField(default=list, verbose_name='Векторное представление')
    
    # Метаданные фрагмента
    word_count = models.PositiveIntegerField(default=0, verbose_name='Количество слов')
    char_count = models.PositiveIntegerField(default=0, verbose_name='Количество символов')
    section_title = models.CharField(max_length=200, blank=True, verbose_name='Название раздела')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Фрагмент документа'
        verbose_name_plural = 'Фрагменты документов'
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
        ]
    
    def __str__(self):
        return f"{self.document.title} - Фрагмент {self.chunk_index}"

class RAGQuery(models.Model):
    """Запросы к RAG системе"""
    QUERY_TYPES = [
        ('search', 'Поиск'),
        ('question', 'Вопрос'),
        ('analysis', 'Анализ'),
        ('recommendation', 'Рекомендация'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Ожидание'),
        ('processing', 'Обработка'),
        ('completed', 'Завершен'),
        ('failed', 'Ошибка'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='Пользователь')
    
    # Запрос
    query_text = models.TextField(verbose_name='Текст запроса')
    query_type = models.CharField(max_length=50, choices=QUERY_TYPES, default='question', verbose_name='Тип запроса')
    
    # Ответ
    response_text = models.TextField(blank=True, verbose_name='Текст ответа')
    confidence_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name='Уверенность ответа'
    )
    
    # Источники
    source_documents = models.ManyToManyField(
        KnowledgeBase,
        through='QuerySource',
        verbose_name='Документы-источники'
    )
    
    # Обработка
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')
    processing_time = models.DurationField(null=True, blank=True, verbose_name='Время обработки')
    error_message = models.TextField(blank=True, verbose_name='Сообщение об ошибке')
    
    # Оценка пользователя
    user_rating = models.PositiveIntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        verbose_name='Оценка пользователя (1-5)'
    )
    user_feedback = models.TextField(blank=True, verbose_name='Отзыв пользователя')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создан')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлен')
    
    class Meta:
        verbose_name = 'RAG запрос'
        verbose_name_plural = 'RAG запросы'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['query_type', '-created_at']),
        ]
    
    def __str__(self):
        return f"Запрос от {self.user.username}: {self.query_text[:100]}..."
    
    def mark_completed(self, response_text, confidence_score=None):
        """Отметить запрос как выполненный"""
        self.response_text = response_text
        self.confidence_score = confidence_score
        self.status = 'completed'
        self.save()

class QuerySource(models.Model):
    """Промежуточная модель для связи запросов и источников"""
    query = models.ForeignKey(RAGQuery, on_delete=models.CASCADE)
    document = models.ForeignKey(KnowledgeBase, on_delete=models.CASCADE)
    relevance_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name='Оценка релевантности'
    )
    chunk_used = models.ForeignKey(
        DocumentChunk, on_delete=models.CASCADE,
        null=True, blank=True,
        verbose_name='Использованный фрагмент'
    )
    
    class Meta:
        verbose_name = 'Источник запроса'
        verbose_name_plural = 'Источники запросов'
        unique_together = ['query', 'document']

class RAGConversation(models.Model):
    """Разговоры/сессии с RAG системой"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='Пользователь')
    title = models.CharField(max_length=200, verbose_name='Название беседы')
    
    # Контекст беседы
    context_data = models.JSONField(default=dict, blank=True, verbose_name='Контекст беседы')
    system_prompt = models.TextField(blank=True, verbose_name='Системный промпт')
    
    # Статистика
    message_count = models.PositiveIntegerField(default=0, verbose_name='Количество сообщений')
    total_tokens = models.PositiveIntegerField(default=0, verbose_name='Общее количество токенов')
    
    # Активность
    is_active = models.BooleanField(default=True, verbose_name='Активна')
    last_activity = models.DateTimeField(auto_now=True, verbose_name='Последняя активность')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создана')
    
    class Meta:
        verbose_name = 'RAG беседа'
        verbose_name_plural = 'RAG беседы'
        ordering = ['-last_activity']
        indexes = [
            models.Index(fields=['user', '-last_activity']),
            models.Index(fields=['is_active', '-last_activity']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.title}"
    
    def add_message(self):
        """Увеличить счетчик сообщений"""
        self.message_count += 1
        self.last_activity = timezone.now()
        self.save(update_fields=['message_count', 'last_activity'])

class ConversationMessage(models.Model):
    """Сообщения в беседах"""
    MESSAGE_TYPES = [
        ('user', 'Пользователь'),
        ('assistant', 'Ассистент'),
        ('system', 'Система'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        RAGConversation, 
        on_delete=models.CASCADE, 
        related_name='messages',
        verbose_name='Беседа'
    )
    
    # Содержимое сообщения
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES, verbose_name='Тип сообщения')
    content = models.TextField(verbose_name='Содержимое')
    
    # Метаданные
    token_count = models.PositiveIntegerField(default=0, verbose_name='Количество токенов')
    response_time = models.DurationField(null=True, blank=True, verbose_name='Время ответа')
    
    # Источники (для сообщений ассистента)
    source_documents = models.ManyToManyField(
        KnowledgeBase,
        blank=True,
        verbose_name='Документы-источники'
    )
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создано')
    
    class Meta:
        verbose_name = 'Сообщение беседы'
        verbose_name_plural = 'Сообщения бесед'
        ordering = ['conversation', 'created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['message_type', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.get_message_type_display()}: {self.content[:100]}..."

class RAGSystemSettings(models.Model):
    """Настройки RAG системы"""
    EMBEDDING_MODELS = [
        ('sentence-transformers/all-MiniLM-L6-v2', 'MiniLM-L6-v2'),
        ('sentence-transformers/all-mpnet-base-v2', 'MPNet-Base-v2'),
        ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'Multilingual-MiniLM-L12-v2'),
    ]
    
    # Модель эмбеддингов
    embedding_model = models.CharField(
        max_length=100,
        choices=EMBEDDING_MODELS,
        default='sentence-transformers/all-MiniLM-L6-v2',
        verbose_name='Модель эмбеддингов'
    )
    embedding_dimensions = models.PositiveIntegerField(default=384, verbose_name='Размерность векторов')
    
    # Параметры поиска
    search_top_k = models.PositiveIntegerField(default=5, verbose_name='Количество топ результатов')
    similarity_threshold = models.FloatField(
        default=0.1,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name='Порог схожести'
    )
    
    # Параметры чанкинга
    chunk_size = models.PositiveIntegerField(default=512, verbose_name='Размер фрагмента')
    chunk_overlap = models.PositiveIntegerField(default=128, verbose_name='Перекрытие фрагментов')
    
    # Настройки генерации
    max_response_tokens = models.PositiveIntegerField(default=1000, verbose_name='Максимум токенов ответа')
    temperature = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(2.0)],
        verbose_name='Температура генерации'
    )
    
    # Кэширование
    enable_caching = models.BooleanField(default=True, verbose_name='Включить кэширование')
    cache_ttl_hours = models.PositiveIntegerField(default=24, verbose_name='TTL кэша (часы)')
    
    # Активные настройки
    is_active = models.BooleanField(default=True, verbose_name='Активные настройки')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Настройки RAG системы'
        verbose_name_plural = 'Настройки RAG системы'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"RAG настройки ({self.embedding_model})"
    
    @classmethod
    def get_active_settings(cls):
        """Получить активные настройки"""
        return cls.objects.filter(is_active=True).first()
