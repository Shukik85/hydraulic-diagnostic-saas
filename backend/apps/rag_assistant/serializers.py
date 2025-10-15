from rest_framework import serializers
from .models import (
    KnowledgeBase, DocumentChunk, RAGQuery, QuerySource, 
    RAGConversation, ConversationMessage, RAGSystemSettings
)

class KnowledgeBaseListSerializer(serializers.ModelSerializer):
    """Сериализатор списка документов базы знаний"""
    category_display = serializers.CharField(source='get_category_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    uploaded_by_name = serializers.CharField(source='uploaded_by.get_full_name', read_only=True)
    
    class Meta:
        model = KnowledgeBase
        fields = [
            'id', 'title', 'category', 'category_display', 'description',
            'document_type', 'document_number', 'version', 'publication_date',
            'status', 'status_display', 'search_count', 'relevance_score',
            'uploaded_by_name', 'created_at', 'updated_at'
        ]

class KnowledgeBaseDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор документа базы знаний"""
    category_display = serializers.CharField(source='get_category_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    uploaded_by_info = serializers.SerializerMethodField()
    chunks_count = serializers.SerializerMethodField()
    content_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = KnowledgeBase
        fields = [
            'id', 'title', 'category', 'category_display', 'description',
            'content', 'content_preview', 'summary', 'keywords',
            'document_type', 'document_number', 'version', 'publication_date',
            'status', 'status_display', 'processing_notes',
            'search_count', 'relevance_score', 'uploaded_by_info',
            'chunks_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'search_count', 'created_at', 'updated_at']
    
    def get_uploaded_by_info(self, obj):
        if obj.uploaded_by:
            return {
                'id': obj.uploaded_by.id,
                'username': obj.uploaded_by.username,
                'full_name': obj.uploaded_by.get_full_name() or obj.uploaded_by.username
            }
        return None
    
    def get_chunks_count(self, obj):
        return obj.chunks.count()
    
    def get_content_preview(self, obj):
        return obj.content[:500] + '...' if len(obj.content) > 500 else obj.content

class DocumentChunkSerializer(serializers.ModelSerializer):
    """Сериализатор фрагментов документов"""
    document_title = serializers.CharField(source='document.title', read_only=True)
    
    class Meta:
        model = DocumentChunk
        fields = [
            'id', 'document', 'document_title', 'content', 'chunk_index',
            'start_position', 'end_position', 'word_count', 'char_count',
            'section_title', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class QuerySourceSerializer(serializers.ModelSerializer):
    """Сериализатор источников запросов"""
    document_info = serializers.SerializerMethodField()
    
    class Meta:
        model = QuerySource
        fields = ['document', 'document_info', 'relevance_score', 'chunk_used']
    
    def get_document_info(self, obj):
        return {
            'id': obj.document.id,
            'title': obj.document.title,
            'category': obj.document.get_category_display(),
            'document_number': obj.document.document_number
        }

class RAGQueryListSerializer(serializers.ModelSerializer):
    """Сериализатор списка RAG запросов"""
    query_type_display = serializers.CharField(source='get_query_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    query_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGQuery
        fields = [
            'id', 'user_name', 'query_text', 'query_preview', 'query_type',
            'query_type_display', 'status', 'status_display', 'confidence_score',
            'user_rating', 'processing_time', 'created_at'
        ]
    
    def get_query_preview(self, obj):
        return obj.query_text[:100] + '...' if len(obj.query_text) > 100 else obj.query_text

class RAGQueryDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор RAG запроса"""
    query_type_display = serializers.CharField(source='get_query_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    user_info = serializers.SerializerMethodField()
    sources = QuerySourceSerializer(source='querysource_set', many=True, read_only=True)
    
    class Meta:
        model = RAGQuery
        fields = [
            'id', 'user_info', 'query_text', 'query_type', 'query_type_display',
            'response_text', 'confidence_score', 'sources', 'status', 'status_display',
            'processing_time', 'error_message', 'user_rating', 'user_feedback',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_user_info(self, obj):
        return {
            'id': obj.user.id,
            'username': obj.user.username,
            'full_name': obj.user.get_full_name() or obj.user.username
        }

class RAGQueryCreateSerializer(serializers.ModelSerializer):
    """Сериализатор создания RAG запроса"""
    class Meta:
        model = RAGQuery
        fields = ['query_text', 'query_type']
    
    def create(self, validated_data):
        request = self.context['request']
        validated_data['user'] = request.user
        return super().create(validated_data)

class ConversationMessageSerializer(serializers.ModelSerializer):
    """Сериализатор сообщений беседы"""
    message_type_display = serializers.CharField(source='get_message_type_display', read_only=True)
    source_documents_info = serializers.SerializerMethodField()
    
    class Meta:
        model = ConversationMessage
        fields = [
            'id', 'message_type', 'message_type_display', 'content',
            'token_count', 'response_time', 'source_documents_info', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_source_documents_info(self, obj):
        return [
            {
                'id': doc.id,
                'title': doc.title,
                'category': doc.get_category_display()
            }
            for doc in obj.source_documents.all()
        ]

class RAGConversationListSerializer(serializers.ModelSerializer):
    """Сериализатор списка бесед"""
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGConversation
        fields = [
            'id', 'user_name', 'title', 'message_count', 'is_active',
            'last_activity', 'last_message', 'created_at'
        ]
    
    def get_last_message(self, obj):
        last_message = obj.messages.order_by('-created_at').first()
        if last_message:
            content = last_message.content
            return content[:100] + '...' if len(content) > 100 else content
        return None

class RAGConversationDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор беседы"""
    user_info = serializers.SerializerMethodField()
    messages = ConversationMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = RAGConversation
        fields = [
            'id', 'user_info', 'title', 'context_data', 'system_prompt',
            'message_count', 'total_tokens', 'is_active', 'last_activity',
            'messages', 'created_at'
        ]
        read_only_fields = ['id', 'message_count', 'total_tokens', 'last_activity', 'created_at']
    
    def get_user_info(self, obj):
        return {
            'id': obj.user.id,
            'username': obj.user.username,
            'full_name': obj.user.get_full_name() or obj.user.username
        }

class RAGSystemSettingsSerializer(serializers.ModelSerializer):
    """Сериализатор настроек RAG системы"""
    embedding_model_display = serializers.CharField(source='get_embedding_model_display', read_only=True)
    
    class Meta:
        model = RAGSystemSettings
        fields = [
            'id', 'embedding_model', 'embedding_model_display', 'embedding_dimensions',
            'search_top_k', 'similarity_threshold', 'chunk_size', 'chunk_overlap',
            'max_response_tokens', 'temperature', 'enable_caching', 'cache_ttl_hours',
            'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

class DocumentUploadSerializer(serializers.Serializer):
    """Сериализатор загрузки документов"""
    file = serializers.FileField(help_text="Файл документа")
    title = serializers.CharField(max_length=500, required=False, help_text="Название документа")
    category = serializers.ChoiceField(choices=KnowledgeBase.CATEGORY_CHOICES, help_text="Категория документа")
    description = serializers.CharField(required=False, help_text="Описание документа")
    document_number = serializers.CharField(max_length=100, required=False, help_text="Номер документа")
    
    def validate_file(self, value):
        # Проверка типа файла
        allowed_types = ['text/plain', 'application/pdf', 'application/msword', 
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        
        if value.content_type not in allowed_types:
            raise serializers.ValidationError(
                "Поддерживаются только файлы: TXT, PDF, DOC, DOCX"
            )
        
        # Проверка размера файла (максимум 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Файл не должен превышать 10MB")
        
        return value

class SearchRequestSerializer(serializers.Serializer):
    """Сериализатор запроса поиска"""
    query = serializers.CharField(help_text="Поисковый запрос")
    category = serializers.ChoiceField(
        choices=KnowledgeBase.CATEGORY_CHOICES,
        required=False,
        help_text="Фильтр по категории"
    )
    top_k = serializers.IntegerField(
        default=5, min_value=1, max_value=20,
        help_text="Количество результатов"
    )
    min_similarity = serializers.FloatField(
        default=0.1, min_value=0.0, max_value=1.0,
        help_text="Минимальная схожесть"
    )

class SearchResultSerializer(serializers.Serializer):
    """Сериализатор результатов поиска"""
    document = KnowledgeBaseListSerializer()
    similarity_score = serializers.FloatField()
    matched_chunk = DocumentChunkSerializer()
    highlighted_content = serializers.CharField()
