from rest_framework import serializers
<<<<<<< HEAD
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import authenticate
from .models import User, UserProfile, UserActivity

class UserRegistrationSerializer(serializers.ModelSerializer):
    """Сериализатор регистрации пользователя"""
    password = serializers.CharField(write_only=True, validators=[validate_password])
=======
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        data['user'] = {
            'id': self.user.id,
            'username': self.user.username,
            'email': self.user.email,
            'role': self.user.role
        }
        return data

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
<<<<<<< HEAD
        fields = [
            'username', 'email', 'password', 'password_confirm',
            'first_name', 'last_name', 'company', 'position', 'phone'
        ]
=======
        fields = ['email', 'username', 'password', 'password_confirm']
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Пароли не совпадают")
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
<<<<<<< HEAD
        password = validated_data.pop('password')
        
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        
        # Создание профиля
        UserProfile.objects.create(user=user)
        
        return user

class UserSerializer(serializers.ModelSerializer):
    """Основной сериализатор пользователя"""
    full_name = serializers.SerializerMethodField()
    systems_count = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 'full_name',
            'company', 'position', 'phone', 'experience_years', 'specialization',
            'email_notifications', 'push_notifications', 'critical_alerts_only',
            'created_at', 'last_activity', 'systems_count', 'reports_generated'
        ]
        read_only_fields = ['id', 'created_at', 'full_name', 'systems_count']
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username
    
    def get_systems_count(self, obj):
        return obj.get_systems_count()

class UserProfileSerializer(serializers.ModelSerializer):
    """Сериализатор профиля пользователя"""
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = [
            'user', 'avatar', 'bio', 'location', 'website',
            'theme', 'language', 'timezone', 'created_at', 'updated_at'
        ]

class UserActivitySerializer(serializers.ModelSerializer):
    """Сериализатор активности пользователя"""
    user_name = serializers.CharField(source='user.username', read_only=True)
    action_display = serializers.CharField(source='get_action_display', read_only=True)
    
    class Meta:
        model = UserActivity
        fields = [
            'id', 'user_name', 'action', 'action_display', 'description',
            'ip_address', 'metadata', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class ChangePasswordSerializer(serializers.Serializer):
    """Сериализатор смены пароля"""
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, validators=[validate_password])
    new_password_confirm = serializers.CharField(required=True)
    
    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError("Новые пароли не совпадают")
        return attrs
    
    def validate_old_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("Неверный текущий пароль")
        return value

class LoginSerializer(serializers.Serializer):
    """Сериализатор входа в систему"""
    email = serializers.EmailField()
    password = serializers.CharField()
    
    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')
        
        if email and password:
            user = authenticate(
                request=self.context.get('request'),
                username=email,
                password=password
            )
            
            if not user:
                raise serializers.ValidationError('Неверные учетные данные')
            
            if not user.is_active:
                raise serializers.ValidationError('Аккаунт деактивирован')
            
            attrs['user'] = user
            return attrs
        else:
            raise serializers.ValidationError('Необходимо указать email и пароль')

class UserStatsSerializer(serializers.Serializer):
    """Сериализатор статистики пользователя"""
    total_systems = serializers.IntegerField()
    active_systems = serializers.IntegerField()
    total_reports = serializers.IntegerField()
    critical_alerts_today = serializers.IntegerField()
    ai_queries_this_month = serializers.IntegerField()
    last_diagnostic_run = serializers.DateTimeField()
=======
        user = User.objects.create_user(**validated_data)
        return user
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
