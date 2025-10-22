from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    # mypy-safe: use generic Manager alias for related managers
    from django.db.models import Manager as RelatedManager
    from apps.diagnostics.models import HydraulicSystem


class User(AbstractUser):
    """Расширенная модель пользователя с типизацией."""

    email: models.EmailField = models.EmailField(unique=True, verbose_name="Email")
    company: models.CharField = models.CharField(max_length=200, blank=True, verbose_name="Компания")
    position: models.CharField = models.CharField(max_length=100, blank=True, verbose_name="Должность")
    phone: models.CharField = models.CharField(max_length=20, blank=True, verbose_name="Телефон")

    experience_years: models.PositiveIntegerField = models.PositiveIntegerField(
        null=True, blank=True, verbose_name="Стаж работы (лет)"
    )
    specialization: models.CharField = models.CharField(
        max_length=100, blank=True, verbose_name="Специализация"
    )

    email_notifications: models.BooleanField = models.BooleanField(
        default=True, verbose_name="Email уведомления"
    )
    push_notifications: models.BooleanField = models.BooleanField(
        default=True, verbose_name="Push уведомления"
    )
    critical_alerts_only: models.BooleanField = models.BooleanField(
        default=False, verbose_name="Только критичные уведомления"
    )

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True, verbose_name="Создан")
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True, verbose_name="Обновлен")
    last_activity: models.DateTimeField = models.DateTimeField(
        default=timezone.now, verbose_name="Последняя активность"
    )

    systems_count: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0, verbose_name="Количество систем"
    )
    reports_generated: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0, verbose_name="Отчетов создано"
    )

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    if TYPE_CHECKING:
        hydraulic_systems: RelatedManager[HydraulicSystem]

    class Meta:
        verbose_name = "Пользователь"
        verbose_name_plural = "Пользователи"
        db_table = "users_user"

    def __str__(self) -> str:
        return f"{self.get_full_name() or self.username} ({self.email})"

    def get_systems_count(self) -> int:
        return self.hydraulic_systems.count()

    def update_last_activity(self) -> None:
        self.last_activity = timezone.now()
        self.save(update_fields=["last_activity"])


class UserProfile(models.Model):
    user: models.OneToOneField = models.OneToOneField(
        User, on_delete=models.CASCADE, verbose_name="Пользователь"
    )
    avatar: models.ImageField = models.ImageField(
        upload_to="avatars/", blank=True, null=True, verbose_name="Аватар"
    )
    bio: models.TextField = models.TextField(blank=True, max_length=500, verbose_name="О себе")
    location: models.CharField = models.CharField(
        max_length=100, blank=True, verbose_name="Местоположение"
    )
    website: models.URLField = models.URLField(blank=True, verbose_name="Веб-сайт")

    theme: models.CharField = models.CharField(
        max_length=20,
        choices=[("light", "Светлая"), ("dark", "Темная"), ("auto", "Автоматически")],
        default="light",
        verbose_name="Тема",
    )
    language: models.CharField = models.CharField(
        max_length=10,
        choices=[("ru", "Русский"), ("en", "English")],
        default="ru",
        verbose_name="Язык",
    )
    timezone: models.CharField = models.CharField(
        max_length=50, default="Europe/Moscow", verbose_name="Часовой пояс"
    )

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Профиль пользователя"
        verbose_name_plural = "Профили пользователей"

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

    user: models.ForeignKey = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Пользователь")
    action: models.CharField = models.CharField(max_length=50, choices=ACTION_TYPES, verbose_name="Действие")
    description: models.TextField = models.TextField(blank=True, verbose_name="Описание")
    ip_address: models.GenericIPAddressField = models.GenericIPAddressField(null=True, blank=True, verbose_name="IP адрес")
    user_agent: models.TextField = models.TextField(blank=True, verbose_name="User Agent")

    metadata: models.JSONField = models.JSONField(default=dict, blank=True, verbose_name="Метаданные")

    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True, verbose_name="Время")

    class Meta:
        verbose_name = "Активность пользователя"
        verbose_name_plural = "Активность пользователей"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["action", "-created_at"]),
        ]

    def __str__(self) -> str:
        # mypy-safe string conversion
        return f"{str(getattr(self.user, 'username', ''))} - {str(self.get_action_display())} ({self.created_at})"
