from django.contrib.auth import get_user_model
from rest_framework import filters, generics, permissions, status, viewsets, mixins
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from .models import UserActivity, UserProfile
from .serializers import (
    ChangePasswordSerializer,
    CustomTokenObtainPairSerializer,
    UserActivitySerializer,
    UserCreateSerializer,
    UserDetailSerializer,
    UserProfileSerializer,
    UserStatsSerializer,
)

User = get_user_model()


class CustomTokenObtainPairView(TokenObtainPairView):
    """Выдача JWT токенов."""

    serializer_class = CustomTokenObtainPairSerializer
    permission_classes = [permissions.AllowAny]


class UserRegistrationView(generics.CreateAPIView):
    """Регистрация нового пользователя."""

    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    permission_classes = [permissions.AllowAny]

    def perform_create(self, serializer):
        user = serializer.save()
        # Логирование
        RefreshToken.for_user(user)
        UserActivity.objects.create(
            user=user,
            action="login",
            description="Регистрация",
            ip_address=self.get_client_ip(),
            user_agent=self.request.META.get("HTTP_USER_AGENT", ""),
        )

    def get_client_ip(self):
        xff = self.request.META.get("HTTP_X_FORWARDED_FOR")
        return xff.split(",")[0] if xff else self.request.META.get("REMOTE_ADDR")


class UserViewSet(viewsets.GenericViewSet, mixins.RetrieveModelMixin, mixins.UpdateModelMixin):
    """Просмотр и обновление данных пользователя."""

    queryset = User.objects.all()
    serializer_class = UserDetailSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = "pk"

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated])
    def change_password(self, request, pk=None):
        """Сменить пароль."""
        user = self.get_object()
        serializer = ChangePasswordSerializer(data=request.data, context={"user": user})
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"detail": "Пароль изменён"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["get"], permission_classes=[permissions.IsAuthenticated])
    def stats(self, request, pk=None):
        """Статистика пользователя."""
        user = self.get_object()
        data = UserStatsSerializer(user).data
        return Response(data)


class UserProfileViewSet(viewsets.ModelViewSet):
    """CRUD для профиля пользователя."""

    queryset = UserProfile.objects.select_related("user").all()
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter]
    search_fields = ["user__username", "location"]


class UserActivityViewSet(viewsets.ReadOnlyModelViewSet):
    """Логи активности пользователей."""

    queryset = UserActivity.objects.select_related("user").all().order_by("-created_at")
    serializer_class = UserActivitySerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["created_at"]
    ordering = ["-created_at"]

    def list(self, request, *args, **kwargs):
        # По умолчанию возвращаем активность текущего пользователя
        self.queryset = self.queryset.filter(user=request.user)
        return super().list(request, *args, **kwargs)
