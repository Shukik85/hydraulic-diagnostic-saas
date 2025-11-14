# Background Drift & AB Test Runner for diagnosis_service

## Описание
Фоновая задача для:
- Периодического мониторинга drift всех моделей сервиса (gnn/rag)
- Оценки активных A/B тестов и автоматического принятия решения (promote/rollback)
- Логирования отчётов о состоянии и алертов

---

## drift_ab_runner.py
```python
import asyncio
import logging
from diagnosis_service.mlops import model_registry, ab_test_manager, get_drift_detector

logger = logging.getLogger("mlops.runner")

async def periodic_drift_and_ab(interval_sec: int = 60):
    """Периодически мониторит drift/A-B тесты для всех production моделей"""
    while True:
        logger.info("[MLOps] Starting drift & A/B test scan...")

        # Проверить drift по всем champion моделям
        for model_type in ["gnn", "rag"]:
            curr = model_registry.get_champion(model_type)
            detector = get_drift_detector(model_type, curr.version)
            score = detector.detect_drift()
            logger.info(f"[MLOps] Drift {model_type}/{curr.version} score: {score:.3f}")
            if score > 0.3:
                logger.warning(f"[MLOps][ALERT] Model drift detected: {model_type} {curr.version} (score={score:.3f})")
                # Здесь интеграция с alertmanager/email/slack

        # A/B tests: авто-решения
        for test_name in list(ab_test_manager._active_tests.keys()):
            result = ab_test_manager.evaluate_test(test_name)
            logger.info(f"[MLOps] AB Test {test_name} decision: {result['decision']} | {result['recommendation']}")
            if result['decision'] in ('promote', 'rollback'):
                ab_test_manager.finalize_test(test_name, result['decision'])
                logger.info(f"[MLOps] AB Test {test_name} auto-finalized: {result['decision']}")

        await asyncio.sleep(interval_sec)

# Для запуска (FastAPI/Celery/отдельный process)
# asyncio.create_task(periodic_drift_and_ab(60))
```

## Checklist
- [x] Мониторинг drift по всем активным champion моделям
- [x] Оценка и auto-promote/rollback активных A/B тестов
- [x] Интеграция с логированием/алертами (расширяется под email/slack)
- [x] Интеграция с diagnosis_service.mlops
