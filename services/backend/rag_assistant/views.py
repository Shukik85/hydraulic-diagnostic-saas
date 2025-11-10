from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from asgiref.sync import async_to_sync
import structlog

from services.rag_client import get_rag_client
from .serializers import RagQuerySerializer

logger = structlog.get_logger()

class RagAssistantViewSet(viewsets.ViewSet):
    """API Gateway для RAG-сервиса с аутентификацией через backend."""

    permission_classes = [IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_client = get_rag_client()

    @action(detail=False, methods=["post"])
    def query(self, request):
        """Проксирует запрос к RAG-сервису с проверкой прав пользователя."""
        serializer = RagQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        system_id = serializer.validated_data.get("system_id")
        # TODO: Добавить granular права доступа пользователей к system_id

        user_context = {
            "user_id": request.user.id,
            "username": request.user.username,
        }
        try:
            rag_response = async_to_sync(self.rag_client.query)(
                query_text=serializer.validated_data["query"],
                system_id=system_id,
                user_context=user_context,
                max_results=serializer.validated_data.get("max_results", 3)
            )
            # Логируем успешный мутации
            logger.info("RAG API Success", user_id=request.user.id, rag_response=rag_response)
            return Response(rag_response, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("RAG API Gateway error")
            return Response({"error": "RAG service unavailable", "info": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
