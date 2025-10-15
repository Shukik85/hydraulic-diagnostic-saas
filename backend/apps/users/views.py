from rest_framework import generics
from rest_framework import status, generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import login, logout
from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta

from .models import User, UserProfile, UserActivity
from .serializers import (
    UserRegistrationSerializer, UserSerializer, UserProfileSerializer,
    ChangePasswordSerializer, LoginSerializer, UserActivitySerializer,
    UserStatsSerializer
)


class UserRegistrationView(generics.CreateAPIView):
    """Регистрация нового пользователя"""
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Создание токенов
        refresh = RefreshToken.for_user(user)

        # Логирование активности
        UserActivity.objects.create(
            user=user,
            action='login',
            description='Первый вход после регистрации',
            ip_address=self.get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )

        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'message': 'Пользователь успешно зарегистрирован'
        }, status=status.HTTP_201_CREATED)

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class UserLoginView(APIView):
    """Вход пользователя в систему"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = LoginSerializer(
            data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)

        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)

        # Обновление времени последней активности
        user.update_last_activity()

        # Логирование входа
        UserActivity.objects.create(
            user=user,
            action='login',
            description='Успешный вход в систему',
            ip_address=self.get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )

        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'message': 'Успешный вход'
        })

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class UserProfileView(generics.RetrieveUpdateAPIView):
    """Просмотр и редактирование профиля пользователя"""
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

    def update(self, request, *args, **kwargs):
        response = super().update(request, *args, **kwargs)

        # Логирование изменения профиля
        UserActivity.objects.create(
            user=request.user,
            action='settings_changed',
            description='Обновление профиля пользователя',
            ip_address=self.get_client_ip(request),
            metadata={'updated_fields': list(request.data.keys())}
        )

        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class UserProfileExtendedView(generics.RetrieveUpdateAPIView):
    """Расширенный профиль пользователя"""
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        profile, created = UserProfile.objects.get_or_create(
            user=self.request.user)
        return profile


class ChangePasswordView(APIView):
    """Смена пароля пользователя"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ChangePasswordSerializer(
            data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)

        user = request.user
        user.set_password(serializer.validated_data['new_password'])
        user.save()

        # Логирование смены пароля
        UserActivity.objects.create(
            user=user,
            action='settings_changed',
            description='Смена пароля',
            ip_address=self.get_client_ip(request),
        )

        return Response({'message': 'Пароль успешно изменен'})

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class UserActivityView(generics.ListAPIView):
    """История активности пользователя"""
    serializer_class = UserActivitySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return UserActivity.objects.filter(user=self.request.user)


class UserStatsView(APIView):
    """Статистика пользователя"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user

        # Подсчет систем
        from apps.diagnostics.models import HydraulicSystem, DiagnosticReport, SensorData

        total_systems = HydraulicSystem.objects.filter(owner=user).count()
        active_systems = HydraulicSystem.objects.filter(
            owner=user,
            status='active'
        ).count()

        # Подсчет отчетов
        total_reports = DiagnosticReport.objects.filter(
            system__owner=user).count()

        # Критические уведомления за сегодня
        today = timezone.now().date()
        critical_alerts_today = SensorData.objects.filter(
            system__owner=user,
            is_critical=True,
            timestamp__date=today
        ).count()

        # AI запросы за месяц
        month_ago = timezone.now() - timedelta(days=30)
        ai_queries_this_month = UserActivity.objects.filter(
            user=user,
            action='ai_query',
            created_at__gte=month_ago
        ).count()

        # Последняя диагностика
        last_diagnostic = UserActivity.objects.filter(
            user=user,
            action='diagnostic_run'
        ).order_by('-created_at').first()

        stats_data = {
            'total_systems': total_systems,
            'active_systems': active_systems,
            'total_reports': total_reports,
            'critical_alerts_today': critical_alerts_today,
            'ai_queries_this_month': ai_queries_this_month,
            'last_diagnostic_run': last_diagnostic.created_at if last_diagnostic else None,
        }

        serializer = UserStatsSerializer(stats_data)
        return Response(serializer.data)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def user_logout(request):
    """Выход пользователя из системы"""
    # Логирование выхода
    UserActivity.objects.create(
        user=request.user,
        action='logout',
        description='Выход из системы',
        ip_address=get_client_ip(request),
    )

    logout(request)
    return Response({'message': 'Успешный выход'})


def get_client_ip(request):
    """Получение IP адреса клиента"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
