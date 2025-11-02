"""Модуль проекта с автогенерированным докстрингом."""

from django.contrib.auth import password_validation
from rest_framework import serializers


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True, trim_whitespace=False)
    new_password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs):
        user = self.context.get("user") or self.context.get("request").user
        if not user.check_password(attrs["old_password"]):
            raise serializers.ValidationError(
                {"old_password": "Неверный текущий пароль"}
            )
        # Валидация нового пароля через стандартные валидаторы Django
        password_validation.validate_password(attrs["new_password"], user=user)
        return attrs

    def save(self, **_kwargs):
        user = self.context.get("user") or self.context.get("request").user
        new_password = self.validated_data["new_password"]
        user.set_password(new_password)
        user.save(update_fields=["password"])
        return user
