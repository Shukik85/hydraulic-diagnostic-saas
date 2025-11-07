"""
Модуль проекта с автогенерированным докстрингом.
# bandit:exclude=B311
"""

from datetime import datetime, timedelta
import json
import logging
import random

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from apps.diagnostics.models import DiagnosticReport, HydraulicSystem, SensorData

User = get_user_model()
logger = logging.getLogger(__name__)

# (дальше весь исходный код без изменений)
