from rest_framework import viewsets, status, permissions, filters
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Document, RagSystem, RagQueryLog
from .serializers import DocumentSerializer, RagSystemSerializer, RagQueryLogSerializer
from .rag_service import RagAssistant

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'language']
    ordering = ['-created_at']

class RagSystemViewSet(viewsets.ModelViewSet):
    queryset = RagSystem.objects.all()
    serializer_class = RagSystemSerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=True, methods=['post'])
    def index(self, request, pk=None):
        system = self.get_object()
        assistant = RagAssistant(system)
        docs = Document.objects.filter(metadata__rag_system=system.id)
        for doc in docs:
            assistant.index_document(doc)
        return Response({'status': 'indexed'}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def query(self, request, pk=None):
        system = self.get_object()
        text = request.data.get('query')
        assistant = RagAssistant(system)
        answer = assistant.answer(text)
        log = RagQueryLog.objects.create(system=system, query_text=text, response_text=answer)
        return Response({'answer': answer}, status=status.HTTP_200_OK)

class RagQueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RagQueryLog.objects.select_related('system', 'document').all()
    serializer_class = RagQueryLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['timestamp']
    ordering = ['-timestamp']
