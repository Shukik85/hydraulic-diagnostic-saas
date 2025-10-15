from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg
from django.utils import timezone
from datetime import timedelta
import logging

from .models import (
    KnowledgeBase, DocumentChunk, RAGQuery, RAGConversation, 
    ConversationMessage, RAGSystemSettings
)
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
            
            # Логирование загрузки документа
            UserActivity.objects.create(
                user=request.user,
                action='document_uploaded',
                description=f'Загружен документ: {document.title}',
                metadata={'document_id': str(document.id), 'category': document.category}
            )
            
            return Response({
                'message': 'Документ успешно загружен и обработан',
                'document': KnowledgeBaseDetailSerializer(document).data
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки документа: {e}")
            return Response({
                'error': 'Ошибка при обработке документа',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def search(self, request):
        """Семантический поиск по базе знаний"""
        serializer = SearchRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            rag_service = RAGService()
            search_results = rag_service.search_documents(
                query=serializer.validated_data['query'],
                category=serializer.validated_data.get('category'),
                top_k=serializer.validated_data.get('top_k', 5),
                min_similarity=serializer.validated_data.get('min_similarity', 0.1)
            )
            
            # Логирование поиска
            UserActivity.objects.create(
                user=request.user,
                action='document_search',
                description=f'Поиск: {serializer.validated_data["query"][:100]}...',
                metadata={
                    'query': serializer.validated_data['query'],
                    'results_count': len(search_results)
                }
            )
            
            # Увеличить счетчики поиска для найденных документов
            document_ids = [result['document'].id for result in search_results]
            KnowledgeBase.objects.filter(id__in=document_ids).update(
                search_count=models.F('search_count') + 1
            )
            
            return Response({
                'query': serializer.validated_data['query'],
                'results': SearchResultSerializer(search_results, many=True).data,
                'total_found': len(search_results)
            })
            
        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            return Response({
                'error': 'Ошибка при поиске документов',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Переобработка документа"""
        document = self.get_object()
        
        try:
            processor = DocumentProcessor()
            processor.reprocess_document(document)
            
            return Response({
                'message': 'Документ успешно переобработан'
            })
            
        except Exception as e:
            logger.error(f"Ошибка переобработки документа {document.id}: {e}")
            return Response({
                'error': 'Ошибка при переобработке документа',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def categories_stats(self, request):
        """Статистика по категориям документов"""
        stats = KnowledgeBase.objects.filter(status='active').values('category').annotate(
            count=Count('id'),
            avg_relevance=Avg('relevance_score'),
            total_searches=models.Sum('search_count')
        ).order_by('-count')
        
        return Response({
            'categories': [
                {
                    'category': item['category'],
                    'category_display': dict(KnowledgeBase.CATEGORY_CHOICES)[item['category']],
                    'count': item['count'],
                    'avg_relevance': round(item['avg_relevance'] or 0, 2),
                    'total_searches': item['total_searches'] or 0
                }
                for item in stats
            ]
        })

class RAGQueryViewSet(viewsets.ModelViewSet):
    """ViewSet для RAG запросов"""
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['query_type', 'status']
    search_fields = ['query_text', 'response_text']
    ordering_fields = ['created_at', 'confidence_score', 'user_rating']
    ordering = ['-created_at']
    
    def get_queryset(self):
        return RAGQuery.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return RAGQueryCreateSerializer
        elif self.action == 'list':
            return RAGQueryListSerializer
        return RAGQueryDetailSerializer
    
    def create(self, request, *args, **kwargs):
        """Создание и обработка RAG запроса"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Создание запроса
        query = serializer.save()
        
        try:
            # Обработка запроса через RAG сервис
            rag_service = RAGService()
            response_data = rag_service.process_query(query)
            
            # Обновление запроса с результатами
            query.mark_completed(
                response_text=response_data['response'],
                confidence_score=response_data.get('confidence', 0.0)
            )
            
            # Сохранение источников
            for source_info in response_data.get('sources', []):
                document = KnowledgeBase.objects.get(id=source_info['document_id'])
                query.querysource_set.create(
                    document=document,
                    relevance_score=source_info['relevance_score'],
                    chunk_used_id=source_info.get('chunk_id')
                )
            
            # Логирование
            UserActivity.objects.create(
                user=request.user,
                action='ai_query',
                description=f'AI запрос: {query.query_text[:100]}...',
                metadata={
                    'query_id': str(query.id),
                    'query_type': query.query_type,
                    'confidence': response_data.get('confidence', 0.0)
                }
            )
            
            return Response(
                RAGQueryDetailSerializer(query).data,
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            # Отметить запрос как неудачный
            query.status = 'failed'
            query.error_message = str(e)
            query.save()
            
            logger.error(f"Ошибка обработки RAG запроса {query.id}: {e}")
            return Response({
                'error': 'Ошибка при обработке запроса',
                'query_id': str(query.id),
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def rate(self, request, pk=None):
        """Оценка ответа пользователем"""
        query = self.get_object()
        
        rating = request.data.get('rating')
        feedback = request.data.get('feedback', '')
        
        if not rating or not (1 <= int(rating) <= 5):
            return Response({
                'error': 'Оценка должна быть от 1 до 5'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        query.user_rating = int(rating)
        query.user_feedback = feedback
        query.save()
        
        return Response({
            'message': 'Оценка сохранена',
            'rating': query.user_rating
        })
    
    @action(detail=False, methods=['get'])
    def user_stats(self, request):
        """Статистика запросов пользователя"""
        queryset = self.get_queryset()
        
        stats = {
            'total_queries': queryset.count(),
            'completed_queries': queryset.filter(status='completed').count(),
            'avg_confidence': queryset.filter(
                status='completed',
                confidence_score__isnull=False
            ).aggregate(avg=Avg('confidence_score'))['avg'] or 0.0,
            'avg_rating': queryset.filter(
                user_rating__isnull=False
            ).aggregate(avg=Avg('user_rating'))['avg'] or 0.0,
            'queries_by_type': queryset.values('query_type').annotate(
                count=Count('id')
            ).order_by('-count'),
            'recent_queries': queryset.order_by('-created_at')[:5].values(
                'id', 'query_text', 'created_at', 'status'
            )
        }
        
        return Response(stats)

class RAGConversationViewSet(viewsets.ModelViewSet):
    """ViewSet для бесед с RAG системой"""
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['is_active']
    search_fields = ['title']
    ordering_fields = ['created_at', 'last_activity', 'message_count']
    ordering = ['-last_activity']
    
    def get_queryset(self):
        return RAGConversation.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return RAGConversationListSerializer
        return RAGConversationDetailSerializer
    
    def perform_create(self, serializer):
        conversation = serializer.save(user=self.request.user)
        
        # Логирование создания беседы
        UserActivity.objects.create(
            user=self.request.user,
            action='conversation_created',
            description=f'Создана беседа: {conversation.title}',
            metadata={'conversation_id': str(conversation.id)}
        )
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Отправка сообщения в беседу"""
        conversation = self.get_object()
        message_content = request.data.get('message', '').strip()
        
        if not message_content:
            return Response({
                'error': 'Сообщение не может быть пустым'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Создание пользовательского сообщения
            user_message = ConversationMessage.objects.create(
                conversation=conversation,
                message_type='user',
                content=message_content
            )
            
            # Обработка сообщения через RAG сервис
            rag_service = RAGService()
            response_data = rag_service.process_conversation_message(
                conversation, message_content
            )
            
            # Создание ответного сообщения
            assistant_message = ConversationMessage.objects.create(
                conversation=conversation,
                message_type='assistant',
                content=response_data['response'],
                response_time=response_data.get('response_time')
            )
            
            # Добавление источников к ответу
            if 'source_documents' in response_data:
                assistant_message.source_documents.set(response_data['source_documents'])
            
            # Обновление счетчиков беседы
            conversation.add_message()
            conversation.add_message()  # +2 (пользователь + ассистент)
            
            return Response({
                'user_message': ConversationMessageSerializer(user_message).data,
                'assistant_message': ConversationMessageSerializer(assistant_message).data
            })
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения в беседе {conversation.id}: {e}")
            return Response({
                'error': 'Ошибка при обработке сообщения',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Архивирование беседы"""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save()
        
        return Response({'message': 'Беседа архивирована'})

class RAGSystemSettingsViewSet(viewsets.ModelViewSet):
    """ViewSet для настроек RAG системы"""
    queryset = RAGSystemSettings.objects.all()
    serializer_class = RAGSystemSettingsSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Активация настроек"""
        settings = self.get_object()
        
        # Деактивировать все остальные настройки
        RAGSystemSettings.objects.update(is_active=False)
        
        # Активировать выбранные настройки
        settings.is_active = True
        settings.save()
        
        return Response({
            'message': 'Настройки активированы',
            'settings': self.get_serializer(settings).data
        })
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Получить активные настройки"""
        active_settings = RAGSystemSettings.get_active_settings()
        
        if active_settings:
            return Response(self.get_serializer(active_settings).data)
        else:
            return Response({
                'message': 'Активные настройки не найдены'
            }, status=status.HTTP_404_NOT_FOUND)

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
        searches=models.Sum('search_count')
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
