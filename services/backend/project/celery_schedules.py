"""
Celery Beat Schedule Configuration for Sensor Data Collection.

Defines periodic tasks for:
- Sensor node polling every 10 seconds
- Health checks every 2 minutes  
- Data cleanup daily at 2:00 AM
"""

from celery.schedules import crontab
from django.conf import settings

# Sensor polling interval (configurable via settings)
SENSOR_POLL_INTERVAL = getattr(settings, 'SENSOR_POLL_INTERVAL_SECONDS', 10)

# Celery Beat Schedule
CELERY_BEAT_SCHEDULE = {
    # Main sensor data collection - runs every 10 seconds
    'poll-sensor-nodes': {
        'task': 'backend.apps.sensors.tasks.poll_sensor_nodes',
        'schedule': SENSOR_POLL_INTERVAL,
        'options': {
            'expires': SENSOR_POLL_INTERVAL + 5,  # Task expires if not run within interval + 5s
            'retry': False,  # Don't retry if task times out
        },
    },
    
    # Sensor health monitoring - runs every 2 minutes
    'sensor-health-check': {
        'task': 'backend.apps.sensors.tasks.sensor_health_check',
        'schedule': 120.0,  # 2 minutes
        'options': {
            'expires': 60,  # Expire after 1 minute
        },
    },
    
    # Data cleanup - runs daily at 2:00 AM
    'cleanup-old-sensor-readings': {
        'task': 'backend.apps.sensors.tasks.cleanup_old_readings',
        'schedule': crontab(hour=2, minute=0),  # 02:00 daily
        'options': {
            'expires': 3600,  # 1 hour to complete
        },
    },
    
    # Optional: TimescaleDB maintenance - runs weekly on Sunday 3:00 AM
    'timescaledb-maintenance': {
        'task': 'backend.apps.sensors.tasks.timescaledb_maintenance',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Sunday 03:00
        'options': {
            'expires': 7200,  # 2 hours to complete
        },
    },
}

# Additional configuration for development/testing
if settings.DEBUG:
    # In development, poll less frequently to reduce noise
    CELERY_BEAT_SCHEDULE['poll-sensor-nodes']['schedule'] = 30  # Every 30 seconds
    
    # Add debug task for testing
    CELERY_BEAT_SCHEDULE['debug-sensor-status'] = {
        'task': 'backend.apps.sensors.tasks.debug_sensor_status',
        'schedule': 60.0,  # Every minute in debug mode
    }
