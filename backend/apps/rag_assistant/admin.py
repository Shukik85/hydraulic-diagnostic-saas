from django.contrib import admin
from django.utils.html import format_html
from .models import (
    KnowledgeBase, DocumentChunk, RAGQuery, QuerySource,
    RAGConversation, ConversationMessage, RAGSystemSettings
)

@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'category', 'document_number', 'status', 
        'search_count', 'relevance_score', 'created_at'
    ]
    list_filter = ['category', 'status', 'document_type', 'created_at']
    search_fields = ['title', 'description', 'document_number', 'keywords']
    readonly_fields = ['search_count', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Основная информация', {
            'fields': ('title', 'category', 'description', 'status')
        }),
        ('Метаданные документа', {
            'fields': ('document_type', 'document_number', 'version', 'publication_date')
        }),
        ('Содержимое', {
            'fields': ('content', 'summary', 'keywords'),
            'classes': ('collapse',)
        }),
        ('Статистика', {
            'fields': ('search_count', 'relevance_score', 'uploaded_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('uploaded_by')

@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'word_count', 'char_count', 'section_title']
    list_filter = ['document__category', 'created_at']
    search_fields = ['document__title', 'content', 'section_title']
    ordering = ['document', 'chunk_index']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('document')

@admin.register(RAGQuery)
class RAGQueryAdmin(admin.ModelAdmin):
    list_display = [
        'get_query_preview', 'user', 'query_type', 'status', 
        'confidence_score', 'user_rating', 'created_at'
    ]
    list_filter = ['query_type', 'status', 'user_rating', 'created_at']
    search_fields = ['query_text', 'response_text', 'user__username']
    readonly_fields = ['created_at', 'updated_at', 'processing_time']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Запрос', {
            'fields': ('user', 'query_text', 'query_type', 'status')
        }),
        ('Ответ', {
            'fields': ('response_text', 'confidence_score', 'processing_time', 'error_message')
        }),
        ('Оценка пользователя', {
            'fields': ('user_rating', 'user_feedback')
        }),
        ('Метаданные', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_query_preview(self, obj):
        return obj.query_text[:100] + '...' if len(obj.query_text) > 100 else obj.query_text
    get_query_preview.short_description = 'Запрос'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(RAGConversation)
class RAGConversationAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'user', 'message_count', 'is_active', 
        'last_activity', 'created_at'
    ]
    list_filter = ['is_active', 'created_at', 'last_activity']
    search_fields = ['title', 'user__username']
    readonly_fields = ['message_count', 'total_tokens', 'last_activity', 'created_at']
    ordering = ['-last_activity']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(ConversationMessage)
class ConversationMessageAdmin(admin.ModelAdmin):
    list_display = [
        'conversation', 'message_type', 'get_content_preview', 
        'token_count', 'created_at'
    ]
    list_filter = ['message_type', 'created_at']
    search_fields = ['conversation__title', 'content']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    def get_content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    get_content_preview.short_description = 'Содержимое'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('conversation')

@admin.register(RAGSystemSettings)
class RAGSystemSettingsAdmin(admin.ModelAdmin):
    list_display = [
        'embedding_model', 'search_top_k', 'similarity_threshold',
        'chunk_size', 'is_active', 'created_at'
    ]
    list_filter = ['is_active', 'embedding_model', 'created_at']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Модель эмбеддингов', {
            'fields': ('embedding_model', 'embedding_dimensions')
        }),
        ('Параметры поиска', {
            'fields': ('search_top_k', 'similarity_threshold')
        }),
        ('Параметры чанкинга', {
            'fields': ('chunk_size', 'chunk_overlap')
        }),
        ('Параметры генерации', {
            'fields': ('max_response_tokens', 'temperature')
        }),
        ('Кэширование', {
            'fields': ('enable_caching', 'cache_ttl_hours')
        }),
        ('Статус', {
            'fields': ('is_active', 'created_at', 'updated_at')
        }),
    )

# Кастомизация админ панели
admin.site.site_header = 'RAG Ассистент - Админ панель'
admin.site.site_title = 'RAG Ассистент'
admin.site.index_title = 'Управление системой RAG'
