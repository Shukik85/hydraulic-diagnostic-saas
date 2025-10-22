"""RAG assistant models with complete type annotations."""
from __future__ import annotations

from django.db import models
from django.db.models import JSONField


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

    title: models.CharField = models.CharField(max_length=255, verbose_name="Название")
    content: models.TextField = models.TextField(verbose_name="Содержимое")
    format: models.CharField = models.CharField(max_length=10, choices=FORMAT_CHOICES, verbose_name="Формат")
    language: models.CharField = models.CharField(max_length=10, choices=LANGUAGE_CHOICES, verbose_name="Язык")
    metadata: JSONField = JSONField(default=dict, verbose_name="Метаданные")
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["language", "format"]),
            models.Index(fields=["created_at"]),
        ]
        verbose_name = "Документ"
        verbose_name_plural = "Документы"

    def __str__(self) -> str:
        return str(self.title)


class RagSystem(models.Model):
    """Конфигурация RAG-пайплайна."""

    name: models.CharField = models.CharField(max_length=100, unique=True)
    description: models.TextField = models.TextField(blank=True)
    model_name: models.CharField = models.CharField(max_length=100, default="openai/gpt-3.5-turbo")
    index_type: models.CharField = models.CharField(max_length=50, default="faiss")
    index_config: JSONField = JSONField(default=dict, verbose_name="Настройки индексатора")
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "RAG-система"
        verbose_name_plural = "RAG-системы"

    def __str__(self) -> str:
        return str(self.name)


class RagQueryLog(models.Model):
    """Логи запросов и ответов."""

    system: models.ForeignKey = models.ForeignKey(RagSystem, on_delete=models.CASCADE, related_name="logs")
    document: models.ForeignKey = models.ForeignKey(Document, on_delete=models.SET_NULL, null=True, blank=True)
    query_text: models.TextField = models.TextField()
    response_text: models.TextField = models.TextField()
    timestamp: models.DateTimeField = models.DateTimeField(auto_now_add=True, db_index=True)
    metadata: JSONField = JSONField(default=dict)

    class Meta:
        verbose_name = "Лог запроса RAG"
        verbose_name_plural = "Логи запросов RAG"
        indexes = [
            models.Index(fields=["timestamp"]),
            models.Index(fields=["system", "timestamp"]),
        ]

    def __str__(self) -> str:
        return f"{str(self.system)} @ {self.timestamp}"
