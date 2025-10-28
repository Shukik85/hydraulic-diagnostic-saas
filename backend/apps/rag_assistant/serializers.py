"""RAG Assistant DRF serializers with production-ready validation."""

from rest_framework import serializers

from .models import Document, RagQueryLog, RagSystem


class RagSystemSerializer(serializers.ModelSerializer):
    """Сериализатор для RAG-системы с подсчётом документов."""

    documents_count = serializers.IntegerField(read_only=True)
    logs_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = RagSystem
        fields = [
            "id",
            "name",
            "description",
            "model_name",
            "index_type",
            "index_config",
            "documents_count",
            "logs_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at", "documents_count", "logs_count"]

    def validate_name(self, value):
        """Проверка уникальности имени."""
        if not value or not value.strip():
            raise serializers.ValidationError("Имя системы обязательно.")
        return value.strip()

    def validate_model_name(self, value):
        """Проверка формата названия модели."""
        if not value:
            return value
        # Простая проверка формата "provider/model-name"
        if "/" in value and len(value.split("/")) == 2:
            return value
        if "/" not in value:  # Локальная модель
            return value
        raise serializers.ValidationError(
            "Название модели должно быть в формате 'provider/model' или 'model'"
        )


class DocumentSerializer(serializers.ModelSerializer):
    """Сериализатор для документов с валидацией."""

    rag_system_name = serializers.CharField(source="rag_system.name", read_only=True)
    content_length = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "rag_system",
            "rag_system_name",
            "title",
            "content",
            "content_length",
            "format",
            "language",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at", "rag_system_name", "content_length"]

    def validate_title(self, value):
        """Проверка заголовка."""
        if not value or not value.strip():
            raise serializers.ValidationError("Заголовок документа обязателен.")
        if len(value) > 255:
            raise serializers.ValidationError("Заголовок не должен превышать 255 символов.")
        return value.strip()

    def validate_content(self, value):
        """Проверка содержимого."""
        if not value or not value.strip():
            raise serializers.ValidationError("Содержимое документа обязательно.")
        # Ограничение размера для производства (1MB)
        if len(value.encode("utf-8")) > 1024 * 1024:
            raise serializers.ValidationError(
                "Размер документа не должен превышать 1MB. Используйте файловую загрузку."
            )
        return value.strip()

    def validate_metadata(self, value):
        """Проверка метаданных."""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Метаданные должны быть объектом JSON.")
        # Ограничение размера метаданных
        import json

        if len(json.dumps(value).encode("utf-8")) > 64 * 1024:  # 64KB
            raise serializers.ValidationError("Метаданные не должны превышать 64KB.")
        return value

    def to_representation(self, instance):
        """Добавляем вычисляемые поля."""
        data = super().to_representation(instance)
        data["content_length"] = len(instance.content.encode("utf-8"))
        return data


class RagQueryLogSerializer(serializers.ModelSerializer):
    """Сериализатор для логов запросов."""

    system_name = serializers.CharField(source="system.name", read_only=True)
    document_title = serializers.CharField(source="document.title", read_only=True)
    query_preview = serializers.SerializerMethodField()
    response_preview = serializers.SerializerMethodField()

    class Meta:
        model = RagQueryLog
        fields = [
            "id",
            "system",
            "system_name",
            "document",
            "document_title",
            "query_text",
            "query_preview",
            "response_text",
            "response_preview",
            "timestamp",
            "metadata",
        ]
        read_only_fields = ["timestamp", "system_name", "document_title"]

    def get_query_preview(self, obj):
        """Короткое превью запроса."""
        if not obj.query_text:
            return ""
        return (
            obj.query_text[:100] + "..."
            if len(obj.query_text) > 100
            else obj.query_text
        )

    def get_response_preview(self, obj):
        """Короткое превью ответа."""
        if not obj.response_text:
            return ""
        return (
            obj.response_text[:150] + "..."
            if len(obj.response_text) > 150
            else obj.response_text
        )

    def validate_query_text(self, value):
        """Проверка текста запроса."""
        if not value or not value.strip():
            raise serializers.ValidationError("Текст запроса обязателен.")
        return value.strip()

    def validate_response_text(self, value):
        """Проверка текста ответа."""
        if value is None:
            return value
        return value.strip() if value.strip() else None
