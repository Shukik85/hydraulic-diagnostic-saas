import os
import sys

import django
from django.db import IntegrityError
from django.utils import timezone

# Настройка Django окружения
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

django.setup()

# После django.setup() можно безопасно импортировать Django модели
from apps.diagnostics.models import (  # noqa: E402
    HydraulicSystem,
    SensorData,
    SystemComponent,
)


def main():
    print("=== Smoke Test: diagnostics models ===")

    # 1) Создание системы и компонента
    hs = HydraulicSystem.objects.create(
        name="Test System", system_type="industrial", status="active"
    )
    print(f"HydraulicSystem created: {hs.id}")

    comp = SystemComponent.objects.create(system=hs, name="Pressure Pump")
    hs.refresh_from_db()
    print("components_count after create:", hs.components_count)

    # 2) SensorData и last_reading_at
    ts1 = timezone.now()
    SensorData.objects.create(
        system=hs, component=comp, timestamp=ts1, unit="bar", value=100.0
    )
    hs.refresh_from_db()
    print("last_reading_at after first reading:", hs.last_reading_at.isoformat())

    ts2 = timezone.now()
    SensorData.objects.create(
        system=hs, component=comp, timestamp=ts2, unit="bar", value=101.5
    )
    hs.refresh_from_db()
    print("last_reading_at after second reading:", hs.last_reading_at.isoformat())

    # 3) Уникальность компонента в рамках системы
    try:
        SystemComponent.objects.create(system=hs, name="Pressure Pump")
        print("ERROR: duplicate component name allowed (should be unique).")
    except IntegrityError:
        print("OK: duplicate component name blocked by unique constraint.")

    # В другой системе одинаковое имя допустимо
    hs2 = HydraulicSystem.objects.create(
        name="Second System", system_type="industrial", status="active"
    )
    comp2 = SystemComponent.objects.create(system=hs2, name="Pressure Pump")
    print("Component with same name in different system OK:", comp2.id is not None)

    print("=== Smoke Test completed ===")


if __name__ == "__main__":
    main()
