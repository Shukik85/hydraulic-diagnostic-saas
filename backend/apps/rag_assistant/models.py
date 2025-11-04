"""Модели Django для приложения RAG Assistant с полными аннотациями типов."""

from __future__ import annotations

from django.db import models
from django.db.models import JSONField


class RagSystem(models.Model):
    """Конфигурация RAG-пайплайна."""

    name: models.CharField = models.CharField(max_length=100, unique=True)
    description: models.TextField = models.TextField(blank=True)
    model_name: models.CharField = models.CharField(
        max_length=100, default="openai/gpt-3.5-turbo"
    )
    index_type: models.CharField = models.CharField(max_length=50, default="faiss")
    index_config: JSONField = JSONField(
        default=dict, verbose_name="Настройки индексатора"
    )
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "RAG-система"
        verbose_name_plural = "RAG-системы"
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        """Возвращает строковое представление RAG системы.

        Returns:
            Название системы
        """
        return str(self.name)


class Document(models.Model):
    """Хранит исходные документы разных форматов и языков."""

    FORMAT_CHOICES = [
        ("txt", "PlainText"),
        ("pdf", "PDF"),
        ("docx", "Word"),
        ("md", "Markdown"),
    ]
    LANGUAGE_CHOICES = [
        ("en", "English"),
        ("ru", "Russian"),
        ("de", "German"),
    ]

    rag_system: models.ForeignKey = models.ForeignKey(
        RagSystem, on_delete=models.CASCADE, related_name="documents"
    )
    title: models.CharField = models.CharField(max_length=255, verbose_name="Название")
    content: models.TextField = models.TextField(verbose_name="Содержимое")
    format: models.CharField = models.CharField(
        max_length=10, choices=FORMAT_CHOICES, verbose_name="Формат"
    )
    language: models.CharField = models.CharField(
        max_length=10, choices=LANGUAGE_CHOICES, verbose_name="Язык"
    )
    metadata: JSONField = JSONField(default=dict, verbose_name="Метаданные")
    created_at: models.DateTimeField = models.DateTimeField(
        auto_now_add=True, db_index=True
    )
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["rag_system", "language"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["format"]),
        ]
        verbose_name = "Документ"
        verbose_name_plural = "Документы"

    def __str__(self) -> str:
        """Возвращает строковое представление документа.

        Returns:
            Заголовок документа
        """
        return str(self.title)


class RagQueryLog(models.Model):
    """Логи запросов и ответов RAG системы."""

    system: models.ForeignKey = models.ForeignKey(
        RagSystem, on_delete=models.CASCADE, related_name="logs"
    )
    document: models.ForeignKey = models.ForeignKey(
        Document, on_delete=models.SET_NULL, null=True, blank=True
    )
    query_text: models.TextField = models.TextField()
    response_text: models.TextField = models.TextField()
    timestamp: models.DateTimeField = models.DateTimeField(
        auto_now_add=True, db_index=True
    )
    metadata: JSONField = JSONField(default=dict)

    class Meta:
        verbose_name = "Лог запроса RAG"
        verbose_name_plural = "Логи запросов RAG"
        indexes = [
            models.Index(fields=["timestamp"]),
            models.Index(fields=["system", "timestamp"]),
        ]

    def __str__(self) -> str:
        """Возвращает строковое представление лога запроса.

        Returns:
            Строка с названием системы и временем запроса
        """
        return f"{self.system!s} @ {self.timestamp}"
