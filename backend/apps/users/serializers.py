"""Модуль проекта с автогенерированным докстрингом."""

from django.contrib.auth import get_user_model

from rest_framework import serializers

from .models import UserActivity, UserProfile

UserModel = get_user_model()


class UserBasicSerializer(serializers.ModelSerializer):
    """Базовая информация о пользователе."""

    class Meta:
        model = UserModel
        fields = ["id", "username", "email", "first_name", "last_name"]
        read_only_fields = fields


class UserProfileSerializer(serializers.ModelSerializer):
    """Сериализатор профиля пользователя."""

    user = UserBasicSerializer(read_only=True)

    class Meta:
        model = UserProfile
        fields = [
            "user",
            "avatar",
            "bio",
            "location",
            "website",
            "theme",
            "language",
            "timezone",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["user", "created_at", "updated_at"]


class UserCreateSerializer(serializers.ModelSerializer):
    """Сериализатор для регистрации нового пользователя."""

    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = UserModel
        fields = ["email", "username", "password", "first_name", "last_name"]

    def create(self, validated_data):
        user = UserModel.objects.create_user(
            email=validated_data["email"],
            username=validated_data["username"],
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
            password=validated_data["password"],
        )
        return user


class UserDetailSerializer(serializers.ModelSerializer):
    """Полная информация о пользователе."""

    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = UserModel
        fields = [
            "id",
            "email",
            "username",
            "first_name",
            "last_name",
            "company",
            "position",
            "phone",
            "experience_years",
            "specialization",
            "email_notifications",
            "push_notifications",
            "critical_alerts_only",
            "created_at",
            "updated_at",
            "last_activity",
            "systems_count",
            "reports_generated",
            "profile",
        ]
        read_only_fields = [
            "id",
            "email",
            "created_at",
            "updated_at",
            "last_activity",
            "systems_count",
            "reports_generated",
        ]


class UserActivitySerializer(serializers.ModelSerializer):
    """Сериализатор логов активности пользователя."""

    user = UserBasicSerializer(read_only=True)

    class Meta:
        model = UserActivity
        fields = [
            "id",
            "user",
            "action",
            "description",
            "ip_address",
            "user_agent",
            "metadata",
            "created_at",
        ]
        read_only_fields = fields
