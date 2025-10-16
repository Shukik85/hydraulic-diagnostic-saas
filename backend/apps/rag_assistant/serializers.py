from rest_framework import serializers
from .models import Document, RagSystem, RagQueryLog

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'format', 'language', 'metadata', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class RagSystemSerializer(serializers.ModelSerializer):
    class Meta:
        model = RagSystem
        fields = ['id', 'name', 'description', 'model_name', 'index_type', 'index_config', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class RagQueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = RagQueryLog
        fields = ['id', 'system', 'document', 'query_text', 'response_text', 'timestamp', 'metadata']
        read_only_fields = ['id', 'timestamp']
