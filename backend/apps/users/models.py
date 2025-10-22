from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager
    from apps.diagnostics.models import HydraulicSystem


class User(AbstractUser):
    email: models.EmailField = models.EmailField(unique=True, verbose_name="Email")
    company: models.CharField = models.CharField(max_length=200, blank=True, verbose_name="Компания")
    position: models.CharField = models.CharField(max_length=100, blank=True, verbose_name="Должность")
    phone: models.CharField = models.CharField(max_length=20, blank=True, verbose_name="Телефон")

    experience_years: models.PositiveIntegerField = models.PositiveIntegerField(null=True, blank=True)
    specialization: models.CharField = models.CharField(max_length=100, blank=True)

    email_notifications: models.BooleanField = models.BooleanField(default=True)
    push_notifications: models.BooleanField = models.BooleanField(default=True)
    critical_alerts_only: models.BooleanField = models.BooleanField(default=False)

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)
    last_activity: models.DateTimeField = models.DateTimeField(default=timezone.now)

    systems_count: models.PositiveIntegerField = models.PositiveIntegerField(default=0)
    reports_generated: models.PositiveIntegerField = models.PositiveIntegerField(default=0)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    if TYPE_CHECKING:
        hydraulic_systems: RelatedManager[HydraulicSystem]

    class Meta:
        db_table = "users_user"

    def __str__(self) -> str:
        return f"{self.get_full_name() or self.username} ({self.email})"


class UserProfile(models.Model):
    user: models.OneToOneField = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar: models.ImageField = models.ImageField(upload_to="avatars/", blank=True, null=True)
    bio: models.TextField = models.TextField(blank=True, max_length=500)
    location: models.CharField = models.CharField(max_length=100, blank=True)
    website: models.URLField = models.URLField(blank=True)

    theme: models.CharField = models.CharField(max_length=20, choices=[("light", "Светлая"), ("dark", "Темная"), ("auto", "Автоматически")], default="light")
    language: models.CharField = models.CharField(max_length=10, choices=[("ru", "Русский"), ("en", "English")], default="ru")
    timezone: models.CharField = models.CharField(max_length=50, default="Europe/Moscow")

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Профиль {self.user.username}"


class UserActivity(models.Model):
    ACTION_TYPES = [
        ("login", "Вход в систему"),
        ("logout", "Выход из системы"),
        ("system_created", "Создание системы"),
        ("system_updated", "Обновление системы"),
        ("system_deleted", "Удаление системы"),
        ("diagnostic_run", "Запуск диагностики"),
        ("report_generated", "Создание отчета"),
        ("settings_changed", "Изменение настроек"),
        ("ai_query", "Запрос к AI"),
    ]

    user: models.ForeignKey = models.ForeignKey(User, on_delete=models.CASCADE)
    action: models.CharField = models.CharField(max_length=50, choices=ACTION_TYPES)
    description: models.TextField = models.TextField(blank=True)
    ip_address: models.GenericIPAddressField = models.GenericIPAddressField(null=True, blank=True)
    user_agent: models.TextField = models.TextField(blank=True)
    metadata: models.JSONField = models.JSONField(default=dict, blank=True)
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        user_str = str(getattr(self.user, "username", ""))
        action_display = getattr(self, "get_action_display", lambda: str(self.action))()
        return f"{user_str} - {action_display} ({self.created_at})"
