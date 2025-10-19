from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class User(AbstractUser):
    """Расширенная модель пользователя"""

    email = models.EmailField(unique=True, verbose_name="Email")
    company = models.CharField(max_length=200, blank=True, verbose_name="Компания")
    position = models.CharField(max_length=100, blank=True, verbose_name="Должность")
    phone = models.CharField(max_length=20, blank=True, verbose_name="Телефон")

    # Профессиональная информация
    experience_years = models.PositiveIntegerField(
        null=True, blank=True, verbose_name="Стаж работы (лет)"
    )
    specialization = models.CharField(
        max_length=100, blank=True, verbose_name="Специализация"
    )

    # Настройки уведомлений
    email_notifications = models.BooleanField(
        default=True, verbose_name="Email уведомления"
    )
    push_notifications = models.BooleanField(
        default=True, verbose_name="Push уведомления"
    )
    critical_alerts_only = models.BooleanField(
        default=False, verbose_name="Только критичные уведомления"
    )

    # Метаданные
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Создан")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Обновлен")
    last_activity = models.DateTimeField(
        default=timezone.now, verbose_name="Последняя активность"
    )

    # Статистика
    systems_count = models.PositiveIntegerField(
        default=0, verbose_name="Количество систем"
    )
    reports_generated = models.PositiveIntegerField(
        default=0, verbose_name="Отчетов создано"
    )

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    class Meta:
        verbose_name = "Пользователь"
        verbose_name_plural = "Пользователи"
        db_table = "users_user"

    def __str__(self):
        return f"{self.get_full_name() or self.username} ({self.email})"

    def get_systems_count(self):
        """Получить количество систем пользователя"""
        return self.hydraulicsystem_set.count()

    def update_last_activity(self):
        """Обновить время последней активности"""
        self.last_activity = timezone.now()
        self.save(update_fields=["last_activity"])


class UserProfile(models.Model):
    """Дополнительный профиль пользователя"""

    user = models.OneToOneField(
        User, on_delete=models.CASCADE, verbose_name="Пользователь"
    )
    avatar = models.ImageField(
        upload_to="avatars/", blank=True, null=True, verbose_name="Аватар"
    )
    bio = models.TextField(blank=True, max_length=500, verbose_name="О себе")
    location = models.CharField(
        max_length=100, blank=True, verbose_name="Местоположение"
    )
    website = models.URLField(blank=True, verbose_name="Веб-сайт")

    # Настройки интерфейса
    theme = models.CharField(
        max_length=20,
        choices=[
            ("light", "Светлая"),
            ("dark", "Темная"),
            ("auto", "Автоматически"),
        ],
        default="light",
        verbose_name="Тема",
    )
    language = models.CharField(
        max_length=10,
        choices=[
            ("ru", "Русский"),
            ("en", "English"),
        ],
        default="ru",
        verbose_name="Язык",
    )
    timezone = models.CharField(
        max_length=50, default="Europe/Moscow", verbose_name="Часовой пояс"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Профиль пользователя"
        verbose_name_plural = "Профили пользователей"

    def __str__(self):
        return f"Профиль {self.user.username}"


class UserActivity(models.Model):
    """Лог активности пользователей"""

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

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, verbose_name="Пользователь"
    )
    action = models.CharField(
        max_length=50, choices=ACTION_TYPES, verbose_name="Действие"
    )
    description = models.TextField(blank=True, verbose_name="Описание")
    ip_address = models.GenericIPAddressField(
        null=True, blank=True, verbose_name="IP адрес"
    )
    user_agent = models.TextField(blank=True, verbose_name="User Agent")

    # Дополнительные данные в JSON формате
    metadata = models.JSONField(default=dict, blank=True, verbose_name="Метаданные")

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Время")

    class Meta:
        verbose_name = "Активность пользователя"
        verbose_name_plural = "Активность пользователей"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["action", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.get_action_display()} ({self.created_at})"
