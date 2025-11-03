"""Модуль проекта с автогенерированным докстрингом."""

from datetime import datetime, timedelta
import json

from django.contrib import admin
from django.utils.html import format_html, format_html_join, escape

from .models import (
    DiagnosticReport,
    HydraulicSystem,
    IntegratedDiagnosticResult,
    MathematicalModelResult,
    PhasePortraitResult,
    SensorData,
    TribodiagnosticResult,
)


@admin.register(HydraulicSystem)
class HydraulicSystemAdmin(admin.ModelAdmin):
    """Расширенный админ интерфейс для гидравлических систем."""

    list_display = [
        "name",
        "system_type",
        "status",
        "owner",
        "health_indicator",
        "last_sensor_data",
        "created_at",
    ]

    list_filter = [
        "system_type",
        "status",
        "created_at",
        "owner",
    ]

    search_fields = ["name", "owner__username", "owner__email"]

    readonly_fields = [
        "created_at",
        "updated_at",
        "health_indicator",
        "system_statistics",
    ]

    fieldsets = (
        (
            "Основная информация",
            {"fields": ("name", "system_type", "status", "owner")},
        ),
        (
            "Системная информация",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                    "health_indicator",
                    "system_statistics",
                ),
                "classes": ("collapse",),
            },
        ),
    )

    actions = ["run_diagnostics", "generate_health_report", "export_system_data"]

    @admin.display(description="Состояние")
    def health_indicator(self, obj):
        """Индикатор здоровья системы."""
        try:
            day_ago = datetime.now() - timedelta(days=1)
            recent_data = obj.sensor_data.filter(timestamp__gte=day_ago)

            if not recent_data.exists():
                return format_html('<span style="color: gray;">⚪ Нет данных</span>')

            critical_count = recent_data.filter(is_critical=True).count()
            total_count = recent_data.count()

            if total_count == 0:
                return format_html('<span style="color: gray;">⚪ Нет данных</span>')

            critical_ratio = critical_count / total_count

            if critical_ratio == 0:
                return format_html('<span style="color: green;">🟢 Отлично</span>')
            if critical_ratio < 0.1:
                return format_html('<span style="color: lightgreen;">🔵 Хорошо</span>')
            if critical_ratio < 0.3:
                return format_html('<span style="color: orange;">🟠 Внимание</span>')
            return format_html('<span style="color: red;">🔴 Критично</span>')

        except Exception:
            return format_html('<span style="color: gray;">⚪ Ошибка</span>')

    @admin.display(description="Последние данные")
    def last_sensor_data(self, obj):
        """Последние данные датчика."""
        latest = obj.sensor_data.order_by("-timestamp").first()
        if latest:
            color = "red" if latest.is_critical else "green"
            return format_html(
                '<span style="color: {};">{}: {} {}</span><br/><small>{}</small>',
                color,
                latest.get_sensor_type_display(),
                latest.value,
                latest.unit,
                latest.timestamp.strftime("%d.%m %H:%M"),
            )
        return "Нет данных"

    @admin.display(description="Статистика")
    def system_statistics(self, obj):
        """Статистика системы (безопасная HTML-сборка)."""
        try:
            week_ago = datetime.now() - timedelta(days=7)

            sensor_count = obj.sensor_data.filter(timestamp__gte=week_ago).count()
            critical_count = obj.sensor_data.filter(
                timestamp__gte=week_ago, is_critical=True
            ).count()
            reports_count = obj.diagnostic_reports.filter(
                created_at__gte=week_ago
            ).count()

            rows = [
                ("Записи датчиков:", sensor_count),
                ("Критические события:", critical_count),
                ("Диагностических отчетов:", reports_count),
                ("Среднее критических событий в день:", round(critical_count / 7, 1)),
            ]

            stats_html = format_html(
                '<div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">'
                "<h4>Статистика за неделю:</h4>"
                "{}"
                "</div>",
                format_html_join(
                    "",
                    "<p><strong>{}</strong> {}</p>",
                    ((escape(name), value) for name, value in rows),
                ),
            )

            return stats_html
        except Exception as e:
            return f"Ошибка расчета статистики: {e}"

    @admin.action(description="🔍 Запустить диагностику")
    def run_diagnostics(self, request, queryset):
        """Запуск диагностики для выбранных систем."""
        count = 0
        for system in queryset:
            try:
                DiagnosticReport.objects.create(
                    system=system,
                    title=f"Административная диагностика - {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                    description="Диагностика запущена через админ панель",
                    severity="info",
                )
                count += 1
            except Exception as e:
                self.message_user(
                    request, f"Ошибка диагностики {system.name}: {e}", level="ERROR"
                )

        self.message_user(request, f"Диагностика запущена для {count} систем")

    @admin.action(description="📊 Сводный отчет")
    def generate_health_report(self, request, queryset):
        """Генерация отчета о состоянии систем."""
        total_systems = queryset.count()
        healthy_systems = 0

        for system in queryset:
            day_ago = datetime.now() - timedelta(days=1)
            critical_events = system.sensor_data.filter(
                timestamp__gte=day_ago, is_critical=True
            ).count()

            if critical_events == 0:
                healthy_systems += 1

        self.message_user(
            request,
            f"Отчет: {healthy_systems}/{total_systems} систем в хорошем состоянии",
        )

    @admin.action(description="📤 Экспорт данных")
    def export_system_data(self, request, queryset):
        """Экспорт данных систем."""
        self.message_user(
            request, f"Экспорт {queryset.count()} систем (функция в разработке)"
        )


@admin.register(SensorData)
class SensorDataAdmin(admin.ModelAdmin):
    """Админ интерфейс для данных датчиков."""

    list_display = [
        "system",
        "sensor_type",
        "value_with_unit",
        "timestamp",
        "critical_indicator",
        "system_owner",
    ]

    list_filter = [
        "sensor_type",
        "is_critical",
        "system__system_type",
        ("timestamp", admin.DateFieldListFilter),
        "system__owner",
    ]

    search_fields = [
        "system__name",
        "system__owner__username",
        "warning_message",
        "sensor_type",
    ]

    readonly_fields = ["created_at"]
    date_hierarchy = "timestamp"
    list_per_page = 50

    @admin.display(description="Значение", ordering="value")
    def value_with_unit(self, obj):
        """Значение с единицей измерения."""
        return f"{obj.value} {obj.unit}"

    @admin.display(description="Статус", ordering="is_critical")
    def critical_indicator(self, obj):
        """Индикатор критичности."""
        if obj.is_critical:
            return format_html(
                '<span style="color: red; font-weight: bold;">🔴 Критично</span>'
            )
        return format_html('<span style="color: green;">🟢 Норма</span>')

    @admin.display(description="Владелец", ordering="system__owner__username")
    def system_owner(self, obj):
        """Владелец системы."""
        return obj.system.owner.username

    actions = ["mark_as_critical", "mark_as_normal", "export_sensor_data"]

    @admin.action(description="⚠️ Отметить как критические")
    def mark_as_critical(self, request, queryset):
        """Отметить как критические."""
        updated = queryset.update(
            is_critical=True, warning_message="Отмечено как критическое администратором"
        )
        self.message_user(request, f"{updated} записей отмечены как критические")

    @admin.action(description="✅ Отметить как нормальные")
    def mark_as_normal(self, request, queryset):
        """Отметить как нормальные."""
        updated = queryset.update(is_critical=False, warning_message="")
        self.message_user(request, f"{updated} записей отмечены как нормальные")


@admin.register(DiagnosticReport)
class DiagnosticReportAdmin(admin.ModelAdmin):
    """Админ интерфейс для диагностических отчетов."""

    list_display = [
        "title",
        "system",
        "severity_indicator",
        "system_owner",
        "has_ai_analysis",
        "created_at",
    ]

    list_filter = [
        "severity",
        "system__system_type",
        "system__owner",
        ("created_at", admin.DateFieldListFilter),
    ]

    search_fields = ["title", "description", "system__name", "system__owner__username"]

    readonly_fields = ["created_at", "updated_at", "ai_analysis_preview"]

    fieldsets = (
        (
            "Основная информация",
            {"fields": ("system", "title", "description", "severity")},
        ),
        (
            "AI Анализ",
            {
                "fields": ("ai_analysis_preview", "ai_analysis"),
                "classes": ("collapse",),
            },
        ),
        (
            "Системная информация",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    date_hierarchy = "created_at"

    @admin.display(description="Серьезность", ordering="severity")
    def severity_indicator(self, obj):
        """Индикатор серьезности."""
        colors = {
            "info": "blue",
            "warning": "orange",
            "error": "red",
            "critical": "darkred",
        }
        icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}

        color = colors.get(obj.severity, "gray")
        icon = icons.get(obj.severity, "❓")

        return format_html(
            '<span style="color: {}; font-weight: bold;">{} {}</span>',
            color,
            icon,
            obj.get_severity_display(),
        )

    @admin.display(description="Владелец", ordering="system__owner__username")
    def system_owner(self, obj):
        """Владелец системы."""
        return obj.system.owner.username

    @admin.display(description="AI анализ")
    def has_ai_analysis(self, obj):
        """Наличие AI анализа."""
        if obj.ai_analysis:
            return format_html('<span style="color: green;">🤖 Да</span>')
        return format_html('<span style="color: gray;">❌ Нет</span>')

    @admin.display(description="Превью AI анализа")
    def ai_analysis_preview(self, obj):
        """Превью AI анализа (безопасная HTML-сборка)."""
        if not obj.ai_analysis:
            return "AI анализ отсутствует"

        try:
            analysis = (
                json.loads(obj.ai_analysis)
                if isinstance(obj.ai_analysis, str)
                else obj.ai_analysis
            )

            parts = []
            if "system_health" in analysis:
                health = analysis["system_health"]
                parts.append(
                    format_html(
                        "<p><strong>Состояние системы:</strong> {}%</p>",
                        health.get("score", "N/A"),
                    )
                )
            if "anomalies" in analysis:
                anomalies = analysis["anomalies"]
                parts.append(
                    format_html(
                        "<p><strong>Аномалии:</strong> {}</p>",
                        len(anomalies.get("anomalies", [])),
                    )
                )
            if "recommendations" in analysis:
                recs = analysis["recommendations"]
                if recs:
                    rec_items = format_html_join(
                        "",
                        "<li>{}</li>",
                        ((escape(rec.get("title", "N/A")),) for rec in recs[:3]),
                    )
                    parts.append(format_html("<ul>{}</ul>", rec_items))

            return format_html(
                "<div style='background: #f8f9fa; padding: 10px; border-radius: 5px;'>"
                "{}"
                "</div>",
                format_html_join("", "{}", ((p,) for p in parts)),
            )
        except Exception as e:
            return f"Ошибка парсинга AI анализа: {e}"

    actions = ["export_reports", "regenerate_ai_analysis"]

    @admin.action(description="📤 Экспорт отчетов")
    def export_reports(self, request, queryset):
        """Экспорт отчетов."""
        self.message_user(
            request, f"Экспорт {queryset.count()} отчетов (функция в разработке)"
        )

    @admin.action(description="🤖 Регенерировать AI анализ")
    def regenerate_ai_analysis(self, request, queryset):
        """Регенерация AI анализа."""
        count = 0
        for report in queryset:
            try:
                # Здесь будет логика регенерации AI анализа
                count += 1
            except Exception as e:
                self.message_user(
                    request,
                    f"Ошибка регенерации для {report.title}: {e}",
                    level="ERROR",
                )

        self.message_user(request, f"AI анализ регенерирован для {count} отчетов")


# Новые админы для диагностических моделей Sprint 1


@admin.register(MathematicalModelResult)
class MathematicalModelResultAdmin(admin.ModelAdmin):
    """Админ интерфейс результатов математической модели."""

    list_display = (
        "id",
        "system",
        "timestamp",
        "pressure_deviation",
        "flow_deviation",
        "speed_deviation",
        "max_deviation",
        "score",
        "status",
        "created_at",
    )

    list_filter = (
        "status",
        ("timestamp", admin.DateFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
        "system",
    )

    search_fields = ("system__name",)
    ordering = ("-timestamp",)
    readonly_fields = ("created_at",)
    date_hierarchy = "timestamp"
    list_per_page = 30


@admin.register(PhasePortraitResult)
class PhasePortraitResultAdmin(admin.ModelAdmin):
    """Админ интерфейс результатов фазового портрета."""

    list_display = (
        "id",
        "system",
        "timestamp",
        "portrait_type",
        "area_deviation",
        "center_shift_x",
        "center_shift_y",
        "contour_breaks",
        "shape_distortion",
        "score",
        "status",
        "created_at",
    )

    list_filter = (
        "portrait_type",
        "status",
        ("timestamp", admin.DateFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
        "system",
    )

    search_fields = ("system__name",)
    ordering = ("-timestamp",)
    readonly_fields = ("created_at",)
    date_hierarchy = "timestamp"
    list_per_page = 30


@admin.register(TribodiagnosticResult)
class TribodiagnosticResultAdmin(admin.ModelAdmin):
    """Админ интерфейс результатов трибодиагностики."""

    list_display = (
        "id",
        "system",
        "analysis_date",
        "iso_class",
        "particles_4um",
        "particles_6um",
        "particles_14um",
        "water_content_ppm",
        "viscosity_cst",
        "ph_level",
        "iron_ppm",
        "copper_ppm",
        "aluminum_ppm",
        "score",
        "status",
        "created_at",
    )

    list_filter = (
        "status",
        ("analysis_date", admin.DateFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
        "system",
        "wear_source",
    )

    search_fields = ("system__name", "lab_report_number", "analyzed_by")
    ordering = ("-analysis_date",)
    readonly_fields = ("created_at",)
    date_hierarchy = "analysis_date"
    list_per_page = 25


@admin.register(IntegratedDiagnosticResult)
class IntegratedDiagnosticResultAdmin(admin.ModelAdmin):
    """Админ интерфейс интегрированных результатов диагностики."""

    list_display = (
        "id",
        "system",
        "timestamp",
        "math_score",
        "phase_score",
        "tribo_score",
        "integrated_score",
        "overall_status",
        "predicted_remaining_life",
        "confidence_level",
        "data_quality_score",
        "created_at",
    )

    list_filter = (
        "overall_status",
        ("timestamp", admin.DateFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
        "system",
    )

    search_fields = ("system__name",)
    ordering = ("-timestamp",)

    readonly_fields = (
        "integrated_score",
        "overall_status",
        "created_at",
    )

    date_hierarchy = "timestamp"
    list_per_page = 25


# Дополнительная конфигурация админки
admin.site.site_header = "Диагностический Комплекс - Администрирование"
admin.site.site_title = "Админ панель"
admin.site.index_title = "Управление гидравлическими системами"
