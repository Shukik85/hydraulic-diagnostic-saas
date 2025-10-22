from __future__ import annotations

from typing import Any, Dict, Type

from django.contrib.auth import get_user_model
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

UserModel: Type = get_user_model()


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Расширенный JWT: добавляет в payload базовые поля пользователя."""

    @classmethod
    def get_token(cls, user):  # type: ignore[override]
        token = super().get_token(user)
        token["username"] = getattr(user, "username", "")
        token["email"] = getattr(user, "email", "")
        token["first_name"] = getattr(user, "first_name", "")
        token["last_name"] = getattr(user, "last_name", "")
        return token


class UserStatsSerializer(serializers.Serializer):
    systems_count = serializers.IntegerField(read_only=True)
    reports_generated = serializers.IntegerField(read_only=True)
    last_activity = serializers.DateTimeField(read_only=True)

    def to_representation(self, instance: Any) -> Dict[str, Any]:
        return {
            "systems_count": getattr(instance, "systems_count", 0),
            "reports_generated": getattr(instance, "reports_generated", 0),
            "last_activity": getattr(instance, "last_activity", None),
        }
