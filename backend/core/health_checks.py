# core/health_checks.py
# COMPREHENSIVE HEALTH CHECKS ДЛЯ PRODUCTION MONITORING

import time
import logging
from typing import Dict, Any, List
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from decouple import config
import redis

# Celery imports
try:
    from celery.app.control import Inspect
    from core.celery import app as celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Логгер для health checks
health_logger = logging.getLogger('apps')

# Health check константы
HEALTH_CHECK_TIMEOUT = 5  # секунд
HEALTH_CHECK_TOKEN = config('HEALTH_CHECK_ACCESS_TOKEN', default='')


def check_database() -> Dict[str, Any]:
    """
    Проверка состояния базы данных
    """
    start_time = time.time()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            'status': 'healthy',
            'response_time_ms': duration_ms,
            'connection_info': {
                'vendor': connection.vendor,
                'version': getattr(connection, 'pg_version', 'unknown'),
                'database': connection.settings_dict.get('NAME', 'unknown')
            }
        }
        
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        health_logger.error(f"Database health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'response_time_ms': duration_ms
        }


def check_redis() -> Dict[str, Any]:
    """
    Проверка состояния Redis
    """
    start_time = time.time()
    try:
        # Основной кеш
        test_key = f'health_check_{int(time.time())}'
        cache.set(test_key, 'ok', 10)
        result = cache.get(test_key)
        cache.delete(test_key)
        
        if result != 'ok':
            raise Exception("Cache set/get/delete test failed")
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        # Получаем информацию о Redis
        redis_info = {}
        try:
            # Прямое подключение для получения INFO
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            info = r.info()
            redis_info = {
                'version': info.get('redis_version', 'unknown'),
                'mode': info.get('redis_mode', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown')
            }
        except Exception:
            redis_info = {'info': 'unavailable'}
        
        return {
            'status': 'healthy',
            'response_time_ms': duration_ms,
            'redis_info': redis_info
        }
        
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        health_logger.error(f"Redis health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'response_time_ms': duration_ms
        }


def check_celery() -> Dict[str, Any]:
    """
    Проверка состояния Celery рабочих процессов
    """
    start_time = time.time()
    
    if not CELERY_AVAILABLE:
        return {
            'status': 'unavailable',
            'error': 'Celery not installed or configured'
        }
    
    try:
        inspect = Inspect(app=celery_app)
        
        # Проверяем доступные рабочие процессы
        active_workers = inspect.active()
        stats = inspect.stats()
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        if not active_workers:
            return {
                'status': 'degraded',
                'warning': 'No active Celery workers found',
                'response_time_ms': duration_ms,
                'workers': 0
            }
        
        # Подсчитываем статистику
        total_workers = len(stats) if stats else 0
        active_tasks = sum(len(tasks) for tasks in active_workers.values()) if active_workers else 0
        
        return {
            'status': 'healthy',
            'response_time_ms': duration_ms,
            'workers_count': total_workers,
            'active_tasks': active_tasks,
            'worker_stats': stats or {}
        }
        
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        health_logger.error(f"Celery health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'response_time_ms': duration_ms
        }


def check_ai_services() -> Dict[str, Any]:
    """
    Проверка доступности AI сервисов (например, OpenAI)
    """
    start_time = time.time()
    
    try:
        openai_key = getattr(settings, 'AI_SETTINGS', {}).get('OPENAI_API_KEY')
        
        if not openai_key:
            return {
                'status': 'unavailable',
                'warning': 'OpenAI API key not configured'
            }
        
        # Можно добавить простой тест API запрос
        # Но это может быть дорого и медленно
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            'status': 'configured',
            'response_time_ms': duration_ms,
            'openai_configured': bool(openai_key),
            'note': 'API availability not tested to avoid costs'
        }
        
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            'status': 'error',
            'error': str(e),
            'response_time_ms': duration_ms
        }


def get_system_metrics() -> Dict[str, Any]:
    """
    Получение базовых системных метрик
    """
    try:
        import psutil
        
        return {
            'cpu_usage_percent': psutil.cpu_percent(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    except ImportError:
        return {
            'note': 'psutil not installed - system metrics unavailable'
        }
    except Exception as e:
        return {
            'error': f'Failed to get system metrics: {str(e)}'
        }


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """
    Комплексный health check endpoint
    Используется load balancer'ami и мониторингом
    """
    overall_start_time = time.time()
    
    # Проверка авторизации для полного health check
    include_detailed = False
    if HEALTH_CHECK_TOKEN and request.GET.get('token') == HEALTH_CHECK_TOKEN:
        include_detailed = True
    
    # Базовые проверки
    checks = {
        'database': check_database(),
        'redis': check_redis(),
    }
    
    # Дополнительные проверки для авторизованных запросов
    if include_detailed:
        checks.update({
            'celery': check_celery(),
            'ai_services': check_ai_services(),
        })
    
    # Определяем общий статус
    overall_status = 'healthy'
    unhealthy_services = []
    degraded_services = []
    
    for service, check_result in checks.items():
        service_status = check_result.get('status', 'unknown')
        if service_status == 'unhealthy':
            overall_status = 'unhealthy'
            unhealthy_services.append(service)
        elif service_status == 'degraded':
            if overall_status == 'healthy':
                overall_status = 'degraded'
            degraded_services.append(service)
    
    overall_duration_ms = round((time.time() - overall_start_time) * 1000, 2)
    
    # Формируем ответ
    response_data = {
        'status': overall_status,
        'timestamp': time.time(),
        'service': 'hydraulic-diagnostic-saas',
        'version': getattr(settings, 'VERSION', '1.0.0'),
        'environment': 'production' if not settings.DEBUG else 'development',
        'total_response_time_ms': overall_duration_ms,
        'checks': checks
    }
    
    # Добавляем проблемные сервисы
    if unhealthy_services:
        response_data['unhealthy_services'] = unhealthy_services
    if degraded_services:
        response_data['degraded_services'] = degraded_services
    
    # Добавляем системные метрики для detailed check
    if include_detailed:
        response_data['system_metrics'] = get_system_metrics()
        
        # Производительность за последний час
        hourly_key = f"performance_metrics:{time.strftime('%Y-%m-%d:%H')}"
        performance_metrics = {
            'requests_last_hour': cache.get(f"{hourly_key}:requests", 0),
            'avg_response_time_ms': cache.get(f"{hourly_key}:avg_duration", 0),
            'ai_requests_last_hour': cache.get(f"ai_metrics:{time.strftime('%Y-%m-%d:%H')}:ai_requests", 0),
        }
        response_data['performance_metrics'] = performance_metrics
    
    # Определяем HTTP статус код
    if overall_status == 'healthy':
        http_status = status.HTTP_200_OK
    elif overall_status == 'degraded':
        http_status = status.HTTP_200_OK  # Все работает, но есть проблемы
    else:
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return Response(response_data, status=http_status)


@api_view(['GET'])
@permission_classes([AllowAny])
def readiness_check(request):
    """
    Kubernetes readiness probe - быстрая проверка
    Проверяет только критические сервисы
    """
    db_check = check_database()
    redis_check = check_redis()
    
    if db_check['status'] == 'healthy' and redis_check['status'] == 'healthy':
        return JsonResponse({
            'status': 'ready',
            'timestamp': time.time()
        })
    else:
        return JsonResponse({
            'status': 'not_ready',
            'timestamp': time.time(),
            'database': db_check['status'],
            'redis': redis_check['status']
        }, status=503)


@api_view(['GET'])
@permission_classes([AllowAny])
def liveness_check(request):
    """
    Kubernetes liveness probe - просто подтверждает, что Django работает
    """
    return JsonResponse({
        'status': 'alive',
        'timestamp': time.time(),
        'service': 'hydraulic-diagnostic-saas'
    })