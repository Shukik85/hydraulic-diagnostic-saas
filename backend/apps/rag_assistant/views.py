from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg, Sum, F
from django.db import models
from django.utils import timezone
from datetime import timedelta
import logging

from .models import (
    KnowledgeBase, DocumentChunk, RAGQuery, RAGConversation, 
    ConversationMessage, RAGSystemSettings
)
from . import models as local_models
from .serializers import (
    KnowledgeBaseListSerializer, KnowledgeBaseDetailSerializer,
    RAGQueryListSerializer, RAGQueryDetailSerializer, RAGQueryCreateSerializer,
    RAGConversationListSerializer, RAGConversationDetailSerializer,
    ConversationMessageSerializer, RAGSystemSettingsSerializer,
    DocumentUploadSerializer, SearchRequestSerializer, SearchResultSerializer
)
from .rag_service import RAGService, DocumentProcessor
from apps.users.models import UserActivity

logger = logging.getLogger('apps.rag_assistant')


class KnowledgeBaseViewSet(viewsets.ModelViewSet):
    """ViewSet для базы знаний"""
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['category', 'status', 'document_type']
    search_fields = ['title', 'description', 'content', 'keywords']
    ordering_fields = ['title', 'created_at', 'search_count', 'relevance_score']
    ordering = ['-created_at']
    parser_classes = [MultiPartParser, FormParser]
    
    def get_queryset(self):
        return KnowledgeBase.objects.filter(status='active')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return KnowledgeBaseListSerializer
        return KnowledgeBaseDetailSerializer
    
    @action(detail=False, methods=['post'])
    def upload_document(self, request):
        """Загрузка нового документа"""
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            # Обработка загруженного файла
            processor = DocumentProcessor()
            document = processor.process_uploaded_file(
                file=serializer.validated_data['file'],
                title=serializer.validated_data.get('title'),
                category=serializer.validated_data['category'],
                description=serializer.validated_data.get('description', ''),
                document_number=serializer.validated_data.get('document_number', ''),
                uploaded_by=request.user
            )
            
            return Response(
                KnowledgeBaseDetailSerializer(document).data,
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки документа: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def increment_search_count(self, request, pk=None):
        """Увеличить счётчик поисков документа"""
        knowledge_base = self.get_object()
        knowledge_base.search_count = F('search_count') + 1
        knowledge_base.last_searched_at = timezone.now()
        knowledge_base.save(update_fields=['search_count', 'last_searched_at'])
        return Response({'message': 'Счётчик увеличен'})
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Поиск в базе знаний с использованием RAG"""
        serializer = SearchRequestSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        
        query_text = serializer.validated_data['query']
        category = serializer.validated_data.get('category')
        top_k = serializer.validated_data.get('top_k', 5)
        
        try:
            # Использование RAG сервиса для поиска
            rag_service = RAGService()
            results = rag_service.search(
                query=query_text,
                category=category,
                top_k=top_k,
                user=request.user
            )
            
            serializer = SearchResultSerializer(results, many=True)
            return Response(serializer.data)
        
        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            return Response(
                {'error': 'Ошибка при выполнении поиска'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Статистика по базе знаний"""
        queryset = self.filter_queryset(self.get_queryset())
        
        stats = {
            'total_documents': queryset.count(),
            'by_category': queryset.values('category').annotate(
                count=Count('id'),
                total_searches=Sum('search_count')
            ),
            'by_document_type': queryset.values('document_type').annotate(
                count=Count('id')
            ),
            'total_chunks': DocumentChunk.objects.filter(
                knowledge_base__in=queryset
            ).count(),
        }
        
        return Response(stats)


class RAGQueryViewSet(viewsets.ModelViewSet):
    """ViewSet для запросов к RAG системе"""
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['status', 'created_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        return RAGQuery.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return RAGQueryCreateSerializer
        elif self.action == 'list':
            return RAGQueryListSerializer
        return RAGQueryDetailSerializer
    
    def perform_create(self, serializer):
        """Создание нового запроса"""
        serializer.save(user=self.request.user)


class RAGConversationViewSet(viewsets.ModelViewSet):
    """ViewSet для бесед с RAG системой"""
    permission_classes = [permissions.IsAuthenticated]
    ordering = ['-created_at']
    
    def get_queryset(self):
        return RAGConversation.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return RAGConversationListSerializer
        return RAGConversationDetailSerializer
    
    def perform_create(self, serializer):
        """Создание новой беседы"""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Отправка сообщения в беседу"""
        conversation = self.get_object()
        
        if not conversation.is_active:
            return Response(
                {'error': 'Беседа завершена'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = ConversationMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            # Создание пользовательского сообщения
            user_message = ConversationMessage.objects.create(
                conversation=conversation,
                message_type='user',
                content=serializer.validated_data['content']
            )
            
            # Получение ответа от RAG системы
            rag_service = RAGService()
            response = rag_service.generate_response(
                conversation=conversation,
                message=serializer.validated_data['content'],
                user=request.user
            )
            
            # Создание ответного сообщения
            assistant_message = ConversationMessage.objects.create(
                conversation=conversation,
                message_type='assistant',
                content=response['answer'],
                sources=response.get('sources', [])
            )
            
            # Обновление последней активности беседы
            conversation.last_message_at = timezone.now()
            conversation.save(update_fields=['last_message_at'])
            
            return Response({
                'user_message': ConversationMessageSerializer(user_message).data,
                'assistant_message': ConversationMessageSerializer(assistant_message).data
            })
        
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {str(e)}")
            return Response(
                {'error': 'Ошибка при обработке сообщения'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def end_conversation(self, request, pk=None):
        """Завершение беседы"""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save(update_fields=['is_active'])
        return Response({'message': 'Беседа завершена'})


class RAGSystemSettingsViewSet(viewsets.ViewSet):
    """ViewSet для настроек RAG системы"""
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """Получение текущих настроек"""
        try:
            settings = RAGSystemSettings.get_active_settings()
            serializer = RAGSystemSettingsSerializer(settings)
            return Response(serializer.data)
        except RAGSystemSettings.DoesNotExist:
            return Response(
                {'message': 'Активные настройки не найдены'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    def update(self, request, pk=None):
        """Обновление настроек"""
        try:
            settings = RAGSystemSettings.get_active_settings()
            serializer = RAGSystemSettingsSerializer(
                settings,
                data=request.data,
                partial=True
            )
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data)
        except RAGSystemSettings.DoesNotExist:
            return Response(
                {'message': 'Активные настройки не найдены'},
                status=status.HTTP_404_NOT_FOUND
            )


@action(detail=False, methods=['get'])
def rag_system_stats(request):
    """Общая статистика RAG системы"""
    if not request.user.is_authenticated:
        return Response({'error': 'Требуется аутентификация'}, status=401)
    
    # Статистика документов
    total_documents = KnowledgeBase.objects.filter(status='active').count()
    total_chunks = DocumentChunk.objects.count()
    
    # Статистика запросов
    total_queries = RAGQuery.objects.count()
    successful_queries = RAGQuery.objects.filter(status='completed').count()
    
    # Статистика бесед
    total_conversations = RAGConversation.objects.count()
    active_conversations = RAGConversation.objects.filter(is_active=True).count()
    
    # Популярные категории
    popular_categories = KnowledgeBase.objects.filter(
        status='active'
    ).values('category').annotate(
        count=Count('id'),
        searches=Sum('search_count')
    ).order_by('-searches')[:5]
    
    # Активность за последние 30 дней
    month_ago = timezone.now() - timedelta(days=30)
    recent_queries = RAGQuery.objects.filter(created_at__gte=month_ago).count()
    
    stats = {
        'documents': {
            'total': total_documents,
            'total_chunks': total_chunks,
            'avg_chunks_per_doc': round(total_chunks / max(total_documents, 1), 1)
        },
        'queries': {
            'total': total_queries,
            'successful': successful_queries,
            'success_rate': round(successful_queries / max(total_queries, 1) * 100, 1),
            'recent_month': recent_queries
        },
        'conversations': {
            'total': total_conversations,
            'active': active_conversations
        },
        'popular_categories': [
            {
                'category': item['category'],
                'category_display': dict(KnowledgeBase.CATEGORY_CHOICES)[item['category']],
                'documents': item['count'],
                'searches': item['searches'] or 0
            }
            for item in popular_categories
        ]
    }
    
    return Response(stats)
