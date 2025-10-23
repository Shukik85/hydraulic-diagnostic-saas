"""Модуль проекта с автогенерированным докстрингом."""

# apps/rag_assistant/serializers.py
from rest_framework import serializers

from .models import Document, RagQueryLog, RagSystem


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "content",
            "format",
            "language",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class RagSystemSerializer(serializers.ModelSerializer):
    # Добавляем агрегированные поля
    document_count = serializers.SerializerMethodField()
    query_count = serializers.SerializerMethodField()

    class Meta:
        model = RagSystem
        fields = [
            "id",
            "name",
            "description",
            "model_name",
            "index_type",
            "index_config",
            "created_at",
            "updated_at",
            "document_count",
            "query_count",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_document_count(self, obj):
        # Получаем из annotate, если доступно
        return getattr(obj, "document_count", 0)

    def get_query_count(self, obj):
        # Получаем из annotate, если доступно
        return getattr(obj, "query_count", 0)


class RagQueryLogSerializer(serializers.ModelSerializer):
    # Eager loading для связанных объектов
    system_name = serializers.CharField(source="system.name", read_only=True)
    document_title = serializers.CharField(
        source="document.title", read_only=True, allow_null=True
    )

    class Meta:
        model = RagQueryLog
        fields = [
            "id",
            "system",
            "system_name",
            "document",
            "document_title",
            "query_text",
            "response_text",
            "timestamp",
            "metadata",
        ]
        read_only_fields = ["id", "timestamp", "system_name", "document_title"]
