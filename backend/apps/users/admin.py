"""Модуль проекта с автогенерированным докстрингом."""

from typing import Any, ClassVar

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import User, UserActivity, UserProfile


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display: ClassVar[list[str]] = [
        "username",
        "email",
        "get_full_name",
        "company",
        "position",
        "systems_count",
        "is_active",
        "last_activity",
        "created_at",
    ]
    list_filter: ClassVar[list[str]] = [
        "is_active",
        "is_staff",
        "is_superuser",
        "company",
        "email_notifications",
        "created_at",
    ]
    search_fields: ClassVar[list[str]] = [
        "username",
        "email",
        "first_name",
        "last_name",
        "company",
    ]
    ordering: ClassVar[list[str]] = ["-created_at"]

    # Приводим BaseUserAdmin.fieldsets к tuple и объединяем с нашими
    base_fieldsets: tuple = tuple(BaseUserAdmin.fieldsets or ())
    extra_fieldsets: tuple = (
        (
            "Дополнительная информация",
            {
                "fields": (
                    "company",
                    "position",
                    "phone",
                    "experience_years",
                    "specialization",
                )
            },
        ),
        (
            "Настройки уведомлений",
            {
                "fields": (
                    "email_notifications",
                    "push_notifications",
                    "critical_alerts_only",
                )
            },
        ),
        (
            "Статистика",
            {"fields": ("systems_count", "reports_generated", "last_activity")},
        ),
    )

    fieldsets = base_fieldsets + extra_fieldsets

    readonly_fields: ClassVar[list[str]] = [
        "last_activity",
        "created_at",
        "systems_count",
    ]

    @admin.display(description="Полное имя")
    def get_full_name(self, obj: User) -> str:
        return obj.get_full_name() or obj.username


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display: ClassVar[list[str]] = [
        "user",
        "location",
        "theme",
        "language",
        "timezone",
        "updated_at",
    ]
    list_filter: ClassVar[list[str]] = ["theme", "language", "timezone", "created_at"]
    search_fields: ClassVar[list[str]] = [
        "user__username",
        "user__email",
        "location",
        "bio",
    ]
    ordering: ClassVar[list[str]] = ["-updated_at"]

    fieldsets = (
        (
            "Основная информация",
            {"fields": ("user", "avatar", "bio", "location", "website")},
        ),
        ("Настройки интерфейса", {"fields": ("theme", "language", "timezone")}),
    )


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display: ClassVar[list[str]] = [
        "user",
        "action",
        "get_action_display",
        "ip_address",
        "created_at",
    ]
    list_filter: ClassVar[list[str]] = ["action", "created_at"]
    search_fields: ClassVar[list[str]] = [
        "user__username",
        "user__email",
        "description",
        "ip_address",
    ]
    ordering: ClassVar[list[str]] = ["-created_at"]
    readonly_fields: ClassVar[list[str]] = ["created_at"]

    fieldsets = (
        ("Основная информация", {"fields": ("user", "action", "description")}),
        (
            "Техническая информация",
            {"fields": ("ip_address", "user_agent", "metadata")},
        ),
        ("Время", {"fields": ("created_at",)}),
    )

    def has_add_permission(self, request: Any) -> bool:
        return False

    def has_change_permission(self, request: Any, obj: Any | None = None) -> bool:
        return False


admin.site.site_header = "Гидравлическая диагностика - Админ панель"
admin.site.site_title = "Админ панель"
admin.site.index_title = "Управление системой диагностики"
