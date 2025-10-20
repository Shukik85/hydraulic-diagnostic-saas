import asyncio
import json
import logging
from datetime import datetime, timedelta

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer

from .models import HydraulicSystem, SensorData

User = get_user_model()
logger = logging.getLogger(__name__)


class DiagnosticSystemConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer для real-time обновлений диагностических систем
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = None
        self.subscribed_systems = set()
        self.room_group_name = None
        self.heartbeat_task = None

    async def connect(self):
        """Подключение клиента"""
        try:
            logger.info(f"WebSocket подключение от {self.scope['client']}")

            # Принимаем подключение
            await self.accept()

            # Отправляем приветственное сообщение
            await self.send_message(
                {
                    "type": "connection_established",
                    "message": "Подключение к диагностической системе установлено",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Запускаем heartbeat
            self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())

        except Exception as e:
            logger.error(f"Ошибка подключения WebSocket: {e}")
            await self.close()

    async def disconnect(self, close_code):
        """Отключение клиента"""
        logger.info(f"WebSocket отключение: {close_code}")

        # Остановка heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Выход из всех групп
        if self.room_group_name:
            await self.channel_layer.group_discard(
                self.room_group_name, self.channel_name
            )

        # Отписка от систем
        for system_id in list(self.subscribed_systems):
            await self.unsubscribe_from_system(system_id)

    async def receive(self, text_data):
        """Получение сообщения от клиента"""
        try:
            data = json.loads(text_data)
            message_type = data.get("type")

            logger.debug(f"Получено WebSocket сообщение: {message_type}")

            # Маршрутизация сообщений
            handler_map = {
                "auth": self.handle_auth,
                "ping": self.handle_ping,
                "subscribe": self.handle_subscribe,
                "subscribe_system": self.handle_subscribe_system,
                "unsubscribe_system": self.handle_unsubscribe_system,
                "get_system_status": self.handle_get_system_status,
                "system_command": self.handle_system_command,
            }

            handler = handler_map.get(message_type)
            if handler:
                await handler(data)
            else:
                await self.send_error(f"Неизвестный тип сообщения: {message_type}")

        except json.JSONDecodeError:
            await self.send_error("Некорректный JSON")
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            await self.send_error("Ошибка обработки сообщения")

    async def handle_auth(self, data):
        """Аутентификация пользователя"""
        token = data.get("token")

        if not token:
            await self.send_message(
                {"type": "auth_failed", "message": "Токен не предоставлен"}
            )
            return

        # Простая проверка токена (в реальном приложении использовать JWT)
        user = await self.get_user_from_token(token)

        if user and not isinstance(user, AnonymousUser):
            self.user = user

            # Присоединение к пользовательской группе
            self.room_group_name = f"user_{user.id}"
            await self.channel_layer.group_add(self.room_group_name, self.channel_name)

            await self.send_message(
                {
                    "type": "auth_success",
                    "message": "Аутентификация успешна",
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                    },
                }
            )

            logger.info(f"Пользователь {user.username} аутентифицирован в WebSocket")

        else:
            await self.send_message(
                {"type": "auth_failed", "message": "Неверный токен"}
            )

    async def handle_ping(self, data):
        """Обработка ping запроса"""
        await self.send_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()}
        )

    async def handle_subscribe(self, data):
        """Подписка на общие каналы"""
        if not self.user:
            await self.send_auth_required()
            return

        channels = data.get("channels", [])

        for channel in channels:
            if channel in [
                "sensor_data",
                "critical_alerts",
                "diagnostic_results",
                "system_status",
            ]:
                group_name = f"{channel}_{self.user.id}"
                await self.channel_layer.group_add(group_name, self.channel_name)
                logger.debug(f"Подписка на канал: {channel}")

        await self.send_message(
            {
                "type": "subscribed",
                "channels": channels,
                "message": "Подписка оформлена успешно",
            }
        )

    async def handle_subscribe_system(self, data):
        """Подписка на обновления конкретной системы"""
        if not self.user:
            await self.send_auth_required()
            return

        system_id = data.get("system_id")
        if not system_id:
            await self.send_error("ID системы не указан")
            return

        # Проверка доступа к системе
        system = await self.get_user_system(system_id)
        if not system:
            await self.send_error("Система не найдена или доступ запрещен")
            return

        await self.subscribe_to_system(system_id)

        await self.send_message(
            {
                "type": "system_subscribed",
                "system_id": system_id,
                "system_name": system.name,
                "message": "Подписка на систему оформлена",
            }
        )

    async def handle_unsubscribe_system(self, data):
        """Отписка от обновлений системы"""
        system_id = data.get("system_id")
        if system_id:
            await self.unsubscribe_from_system(system_id)

            await self.send_message(
                {
                    "type": "system_unsubscribed",
                    "system_id": system_id,
                    "message": "Отписка от системы выполнена",
                }
            )

    async def handle_get_system_status(self, data):
        """Получение текущего статуса системы"""
        if not self.user:
            await self.send_auth_required()
            return

        system_id = data.get("system_id")
        system = await self.get_user_system(system_id)

        if not system:
            await self.send_error("Система не найдена")
            return

        # Получение последних данных
        latest_sensor_data = await self.get_latest_sensor_data(system_id)
        critical_events = await self.get_recent_critical_events(system_id)

        await self.send_message(
            {
                "type": "system_status",
                "system_id": system_id,
                "status": system.status,
                "latest_sensor_data": latest_sensor_data,
                "critical_events_count": len(critical_events),
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def handle_system_command(self, data):
        """Обработка команд системе"""
        if not self.user:
            await self.send_auth_required()
            return

        system_id = data.get("system_id")
        command = data.get("command")
        data.get("params", {})

        system = await self.get_user_system(system_id)
        if not system:
            await self.send_error("Система не найдена")
            return

        # Обработка команд
        if command == "start_monitoring":
            await self.start_system_monitoring(system_id)
        elif command == "stop_monitoring":
            await self.stop_system_monitoring(system_id)
        elif command == "run_diagnostic":
            await self.trigger_diagnostic(system_id)
        else:
            await self.send_error(f"Неизвестная команда: {command}")

    async def subscribe_to_system(self, system_id):
        """Подписка на систему"""
        group_name = f"system_{system_id}"
        await self.channel_layer.group_add(group_name, self.channel_name)
        self.subscribed_systems.add(system_id)
        logger.debug(f"Подписка на систему {system_id}")

    async def unsubscribe_from_system(self, system_id):
        """Отписка от системы"""
        group_name = f"system_{system_id}"
        await self.channel_layer.group_discard(group_name, self.channel_name)
        self.subscribed_systems.discard(system_id)
        logger.debug(f"Отписка от системы {system_id}")

    async def start_system_monitoring(self, system_id):
        """Запуск мониторинга системы"""
        # Здесь может быть логика запуска мониторинга
        await self.send_message(
            {
                "type": "monitoring_started",
                "system_id": system_id,
                "message": "Мониторинг системы запущен",
            }
        )

    async def stop_system_monitoring(self, system_id):
        """Остановка мониторинга системы"""
        await self.send_message(
            {
                "type": "monitoring_stopped",
                "system_id": system_id,
                "message": "Мониторинг системы остановлен",
            }
        )

    async def trigger_diagnostic(self, system_id):
        """Запуск диагностики системы"""
        # Имитация запуска диагностики
        await self.send_message(
            {
                "type": "diagnostic_started",
                "system_id": system_id,
                "message": "Диагностика системы запущена",
            }
        )

        # Имитация завершения диагностики через несколько секунд
        await asyncio.sleep(3)

        await self.send_message(
            {
                "type": "diagnostic_completed",
                "system_id": system_id,
                "message": "Диагностика системы завершена",
                "result": {
                    "status": "completed",
                    "health_score": 85,
                    "issues_found": 2,
                },
            }
        )

    # Обработчики событий от каналов
    async def sensor_data_update(self, event):
        """Обработка обновления данных датчиков"""
        await self.send_message(
            {
                "type": "sensor_data",
                "system_id": event["system_id"],
                "sensor_type": event["sensor_type"],
                "value": event["value"],
                "unit": event["unit"],
                "timestamp": event["timestamp"],
                "is_critical": event.get("is_critical", False),
                "warning_message": event.get("warning_message", ""),
            }
        )

    async def critical_alert(self, event):
        """Обработка критических предупреждений"""
        await self.send_message(
            {
                "type": "critical_alert",
                "system_id": event["system_id"],
                "system_name": event.get("system_name", ""),
                "alert_type": event["alert_type"],
                "message": event["message"],
                "severity": event.get("severity", "high"),
                "timestamp": event["timestamp"],
                "recommended_actions": event.get("recommended_actions", []),
            }
        )

    async def diagnostic_result(self, event):
        """Обработка результатов диагностики"""
        await self.send_message(
            {
                "type": "diagnostic_result",
                "system_id": event["system_id"],
                "report_id": event.get("report_id"),
                "result": event["result"],
                "timestamp": event["timestamp"],
            }
        )

    async def system_status_change(self, event):
        """Обработка изменения статуса системы"""
        await self.send_message(
            {
                "type": "system_status_change",
                "system_id": event["system_id"],
                "old_status": event["old_status"],
                "new_status": event["new_status"],
                "timestamp": event["timestamp"],
            }
        )

    # Вспомогательные методы
    async def send_message(self, message):
        """Отправка сообщения клиенту"""
        await self.send(text_data=json.dumps(message, ensure_ascii=False))

    async def send_error(self, error_message):
        """Отправка ошибки клиенту"""
        await self.send_message(
            {
                "type": "error",
                "message": error_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def send_auth_required(self):
        """Уведомление о необходимости аутентификации"""
        await self.send_error("Требуется аутентификация")

    async def heartbeat_loop(self):
        """Heartbeat для поддержания соединения"""
        try:
            while True:
                await asyncio.sleep(30)  # Ping каждые 30 секунд
                await self.send_message(
                    {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
                )
        except asyncio.CancelledError:
            logger.debug("Heartbeat остановлен")

    # База данных методы (sync_to_async)
    @database_sync_to_async
    def get_user_from_token(self, token):
        """Получение пользователя по токену"""
        try:
            # Простая проверка токена
            # В реальном приложении здесь должна быть проверка JWT токена
            from django.contrib.auth.models import User

            return User.objects.filter(is_active=True).first()
        except Exception as e:
            logger.error(f"Ошибка проверки токена: {e}")
            return None

    @database_sync_to_async
    def get_user_system(self, system_id):
        """Получение системы пользователя"""
        try:
            if not self.user:
                return None
            return HydraulicSystem.objects.get(id=system_id, owner=self.user)
        except HydraulicSystem.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Ошибка получения системы: {e}")
            return None

    @database_sync_to_async
    def get_latest_sensor_data(self, system_id):
        """Получение последних данных датчиков"""
        try:
            latest_data = []
            sensor_types = ["pressure", "temperature", "flow", "vibration"]

            for sensor_type in sensor_types:
                data = (
                    SensorData.objects.filter(
                        system_id=system_id, sensor_type=sensor_type
                    )
                    .order_by("-timestamp")
                    .first()
                )

                if data:
                    latest_data.append(
                        {
                            "sensor_type": data.sensor_type,
                            "value": data.value,
                            "unit": data.unit,
                            "timestamp": data.timestamp.isoformat(),
                            "is_critical": data.is_critical,
                        }
                    )

            return latest_data
        except Exception as e:
            logger.error(f"Ошибка получения данных датчиков: {e}")
            return []

    @database_sync_to_async
    def get_recent_critical_events(self, system_id, hours=24):
        """Получение недавних критических событий"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            critical_events = SensorData.objects.filter(
                system_id=system_id, is_critical=True, timestamp__gte=cutoff_time
            ).order_by("-timestamp")[:10]

            return [
                {
                    "sensor_type": event.sensor_type,
                    "value": event.value,
                    "unit": event.unit,
                    "timestamp": event.timestamp.isoformat(),
                    "warning_message": event.warning_message,
                }
                for event in critical_events
            ]
        except Exception as e:
            logger.error(f"Ошибка получения критических событий: {e}")
            return []


# Утилиты для отправки WebSocket сообщений из других частей приложения


def send_sensor_data_update(system_id, sensor_data):
    """Отправка обновления данных датчика через WebSocket"""
    channel_layer = get_channel_layer()
    group_name = f"system_{system_id}"

    message = {
        "type": "sensor_data_update",
        "system_id": system_id,
        "sensor_type": sensor_data.sensor_type,
        "value": sensor_data.value,
        "unit": sensor_data.unit,
        "timestamp": sensor_data.timestamp.isoformat(),
        "is_critical": sensor_data.is_critical,
        "warning_message": sensor_data.warning_message,
    }

    async_to_sync(channel_layer.group_send)(group_name, message)


def send_critical_alert(system, alert_type, message, severity="high"):
    """Отправка критического предупреждения"""
    channel_layer = get_channel_layer()
    group_name = f"system_{system.id}"

    alert_message = {
        "type": "critical_alert",
        "system_id": system.id,
        "system_name": system.name,
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "recommended_actions": [],
    }

    async_to_sync(channel_layer.group_send)(group_name, alert_message)


def send_diagnostic_result(system, report):
    """Отправка результата диагностики"""
    channel_layer = get_channel_layer()
    group_name = f"system_{system.id}"

    result_message = {
        "type": "diagnostic_result",
        "system_id": system.id,
        "system_name": system.name,
        "report_id": report.id,
        "result": {
            "title": report.title,
            "severity": report.severity,
            "description": report.description,
        },
        "timestamp": report.created_at.isoformat(),
    }

    async_to_sync(channel_layer.group_send)(group_name, result_message)
