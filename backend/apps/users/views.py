from rest_framework import status, generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import login, logout
from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta
from .models import User, UserProfile, UserActivity
from .serializers import (
    UserRegistrationSerializer, UserSerializer, UserProfileSerializer,
    ChangePasswordSerializer, LoginSerializer, UserActivitySerializer,
    UserStatsSerializer, CustomTokenObtainPairSerializer
)


class CustomTokenObtainPairView(TokenObtainPairView):
    """Кастомный view для получения токенов"""
    serializer_class = CustomTokenObtainPairSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        print(f"Login attempt: {request.data}")  # Для отладки
        try:
            response = super().post(request, *args, **kwargs)
            print(f"Login successful")
            return response
        except Exception as e:
            print(f"Login error: {str(e)}")
            return Response({
                'error': 'Неверные учетные данные'
            }, status=status.HTTP_400_BAD_REQUEST)


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
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        
        # Создание токенов
        refresh = RefreshToken.for_user(user)
        
        # Логирование входа
        UserActivity.objects.create(
            user=user,
            action='login',
            description='Вход в систему',
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )
        
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'message': 'Успешный вход'
        })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """Получение профиля пользователя"""
    return Response({
        'id': request.user.id,
        'username': request.user.username,
        'email': request.user.email,
        'role': request.user.role
    })


def get_client_ip(request):
    """Получение IP адреса клиента"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
