#!/usr/bin/env python3
"""
Push UCI Hydraulic data to ML Inference API for testing

Тестирование ML сервиса на реальных данных UCI:
- Читает cycles_sample_100.parquet
- Конвертирует в SensorDataBatch формат
- Отправляет через /api/v1/predict/batch
- Измеряет latency, анализирует результаты

Usage:
  # Запустить ML сервис в другом терминале:
  cd ml_service && python main.py

  # Затем тестирование:
  python ml_service/scripts/push_to_api.py --cycles 10 --api http://localhost:8001
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import structlog

# Настройка логирования
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MLAPITester:
    """Тестирование ML API с UCI данными."""

    def __init__(self, api_base: str, timeout: int = 30):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.results: list[dict[str, Any]] = []

    async def test_service_health(self) -> bool:
        """Проверка готовности ML сервиса."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Health check
                health_resp = await client.get(f"{self.api_base}/health")
                ready_resp = await client.get(f"{self.api_base}/ready")

                if health_resp.status_code == 200 and ready_resp.status_code == 200:
                    logger.info("ML service is healthy and ready")
                    return True
                else:
                    logger.error(
                        "ML service not ready",
                        health_status=health_resp.status_code,
                        ready_status=ready_resp.status_code,
                    )
                    return False

        except Exception as e:
            logger.error("Failed to check ML service health", error=str(e))
            return False

    def load_uci_data(self, data_path: str, limit_cycles: int) -> pd.DataFrame:
        """Загрузка UCI данных из Parquet."""
        try:
            df = pd.read_parquet(data_path)

            # Ограничиваем количество циклов
            available_cycles = df["cycle"].nunique()
            if limit_cycles > available_cycles:
                logger.warning(
                    "Requested more cycles than available", requested=limit_cycles, available=available_cycles
                )
                limit_cycles = available_cycles

            cycles_to_test = df["cycle"].unique()[:limit_cycles]
            filtered_df = df[df["cycle"].isin(cycles_to_test)]

            logger.info(
                "UCI data loaded",
                total_rows=len(filtered_df),
                cycles=limit_cycles,
                sensors=filtered_df["sensor"].nunique(),
                sensor_types=list(filtered_df["sensor_type"].unique()),
            )

            return filtered_df

        except Exception as e:
            logger.error("Failed to load UCI data", error=str(e), path=data_path)
            raise

    def convert_cycle_to_sensor_batch(self, cycle_df: pd.DataFrame, cycle_id: int) -> dict[str, Any]:
        """Конвертация одного цикла UCI в SensorDataBatch."""
        readings = []

        for _, row in cycle_df.iterrows():
            # Создаем абсолютный timestamp для цикла
            base_time = datetime.now(UTC)
            cycle_timestamp = base_time.replace(
                second=int(row["timestamp"]) % 60,  # ✅ Приводим к диапазону 0-59
                microsecond=int((row["timestamp"] % 1) * 1000000),
            )

            reading = {
                "timestamp": cycle_timestamp.isoformat(),
                "sensor_type": row["sensor_type"],
                "value": float(row["value"]) if hasattr(row["value"], "item") else float(row["value"]),
                "unit": row["unit"],
                "component_id": None,  # UCI данные не содержат component_id
            }
            readings.append(reading)

        return {
            "sensor_data": {
                "system_id": str(cycle_df["system_id"].iloc[0]),
                "readings": readings,
                "metadata": {"source": "UCI_Hydraulic", "cycle": cycle_id, "sensors_count": len(readings)},
            },
            "prediction_type": "anomaly",
            "use_cache": True,
            "ml_models": ["catboost", "xgboost", "random_forest"],  # обновлено поле и состав моделей
        }

    async def test_single_prediction(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Тест одиночного предсказания."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/api/v1/predict", json=request_data, headers={"Content-Type": "application/json"}
                )

                request_time = (time.time() - start_time) * 1000  # ms

                if response.status_code == 200:
                    result = response.json()

                    return {
                        "success": True,
                        "request_time_ms": request_time,
                        "api_processing_time_ms": result.get("total_processing_time_ms", 0),
                        "ensemble_score": result.get("ensemble_score", 0),
                        "severity": result.get("prediction", {}).get("severity", "unknown"),
                        "confidence": result.get("prediction", {}).get("confidence", 0),
                        "models_used": len(result.get("ml_predictions", [])),
                        "features_extracted": result.get("features_extracted", 0),
                        "cache_hit": result.get("cache_hit", False),
                    }
                else:
                    logger.error("API request failed", status_code=response.status_code, response=response.text[:200])
                    return {
                        "success": False,
                        "request_time_ms": request_time,
                        "error": f"HTTP {response.status_code}: {response.text[:100]}",
                    }

        except Exception as e:
            request_time = (time.time() - start_time) * 1000
            logger.error("Single prediction failed", error=str(e))
            return {"success": False, "request_time_ms": request_time, "error": str(e)}

    async def test_batch_prediction(self, requests_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Тест пакетного предсказания."""
        start_time = time.time()

        batch_request = {"requests": requests_data, "parallel_processing": True}

        try:
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                response = await client.post(
                    f"{self.api_base}/api/v1/predict/batch",
                    json=batch_request,
                    headers={"Content-Type": "application/json"},
                )

                request_time = (time.time() - start_time) * 1000  # ms

                if response.status_code == 200:
                    result = response.json()

                    return {
                        "success": True,
                        "batch_size": len(requests_data),
                        "request_time_ms": request_time,
                        "api_processing_time_ms": result.get("total_processing_time_ms", 0),
                        "successful_predictions": result.get("successful_predictions", 0),
                        "failed_predictions": result.get("failed_predictions", 0),
                        "results": result.get("results", []),
                    }
                else:
                    return {
                        "success": False,
                        "batch_size": len(requests_data),
                        "request_time_ms": request_time,
                        "error": f"HTTP {response.status_code}: {response.text[:100]}",
                    }

        except Exception as e:
            request_time = (time.time() - start_time) * 1000
            logger.error("Batch prediction failed", error=str(e))
            return {
                "success": False,
                "batch_size": len(requests_data),
                "request_time_ms": request_time,
                "error": str(e),
            }

    def analyze_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Анализ результатов тестирования."""
        if not results:
            return {"error": "No results to analyze"}

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        if not successful:
            return {
                "total_tests": len(results),
                "success_rate": 0.0,
                "errors": [r.get("error", "Unknown error") for r in failed],
            }

        # Анализ latency
        request_times = [r["request_time_ms"] for r in successful]
        api_times = [r.get("api_processing_time_ms", 0) for r in successful if r.get("api_processing_time_ms")]

        # Анализ предсказаний
        scores = [r.get("ensemble_score", 0) for r in successful if r.get("ensemble_score") is not None]
        severities = [r.get("severity", "unknown") for r in successful]
        confidences = [r.get("confidence", 0) for r in successful if r.get("confidence")]

        severity_counts: dict[str, int] = {}
        for sev in severities:
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        import numpy as np

        analysis = {
            "total_tests": len(results),
            "successful_tests": len(successful),
            "failed_tests": len(failed),
            "success_rate": len(successful) / len(results) * 100,
            "latency_stats": {
                "request_time_p50_ms": float(np.percentile(request_times, 50)) if request_times else 0,
                "request_time_p90_ms": float(np.percentile(request_times, 90)) if request_times else 0,
                "request_time_p99_ms": float(np.percentile(request_times, 99)) if request_times else 0,
                "api_time_p90_ms": float(np.percentile(api_times, 90)) if api_times else 0,
                "target_p90_ms": 100,
                "meets_target": (np.percentile(request_times, 90) < 100) if request_times else False,
            },
            "prediction_stats": {
                "ensemble_score_mean": float(np.mean(scores)) if scores else 0,
                "ensemble_score_std": float(np.std(scores)) if scores else 0,
                "confidence_mean": float(np.mean(confidences)) if confidences else 0,
                "severity_distribution": severity_counts,
            },
            "errors": [r.get("error", "Unknown") for r in failed] if failed else [],
        }

        return analysis


async def main():
    parser = argparse.ArgumentParser(description="Test ML API with UCI data")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles to test")
    parser.add_argument("--api", type=str, default="http://localhost:8001", help="ML API base URL")
    parser.add_argument(
        "--data", type=str, default="data/uci_hydraulic/cycles_sample_100.parquet", help="UCI data file"
    )
    parser.add_argument("--batch", action="store_true", help="Use batch API instead of single predictions")
    parser.add_argument("--report", type=str, default="reports/uci_test_report.json", help="Save report to file")

    args = parser.parse_args()

    # Проверяем наличие данных
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("UCI data file not found", path=str(data_path))
        logger.info("Run first: python ml_service/scripts/fetch_uci_hydraulic.py --limit 100")
        return

    # Создаем тестер
    tester = MLAPITester(args.api)

    # Проверяем готовность сервиса
    if not await tester.test_service_health():
        logger.error("ML service is not ready. Start it with: cd ml_service && python main.py")
        return

    # Загружаем данные
    logger.info("Loading UCI data", path=str(data_path), cycles=args.cycles)
    df = tester.load_uci_data(str(data_path), args.cycles)

    # Подготавливаем запросы по циклам
    requests_data: list[dict[str, Any]] = []

    for cycle_id in df["cycle"].unique()[: args.cycles]:
        cycle_df = df[df["cycle"] == cycle_id]
        request = tester.convert_cycle_to_sensor_batch(cycle_df, cycle_id)
        requests_data.append(request)

    logger.info("Prepared requests", count=len(requests_data))

    # Выполняем тестирование
    results: list[dict[str, Any]] = []

    if args.batch:
        logger.info("Running batch prediction test")
        batch_result = await tester.test_batch_prediction(requests_data)
        results.append(batch_result)

        # Если batch успешен, извлекаем индивидуальные результаты
        if batch_result.get("success") and batch_result.get("results"):
            for _i, individual_result in enumerate(batch_result["results"]):  # i → _i
                if isinstance(individual_result, dict) and "ensemble_score" in individual_result:
                    results.append(
                        {
                            "success": True,
                            "request_time_ms": batch_result["request_time_ms"] / len(requests_data),
                            "api_processing_time_ms": individual_result.get("total_processing_time_ms", 0),
                            "ensemble_score": individual_result.get("ensemble_score", 0),
                            "severity": individual_result.get("prediction", {}).get("severity", "unknown"),
                            "confidence": individual_result.get("prediction", {}).get("confidence", 0),
                            "models_used": len(individual_result.get("ml_predictions", [])),  # обновлено поле
                            "features_extracted": individual_result.get("features_extracted", 0),
                            "cache_hit": individual_result.get("cache_hit", False),
                        }
                    )
    else:
        logger.info("Running individual prediction tests")
        for i, request in enumerate(requests_data):
            logger.info(f"Testing cycle {i + 1}/{len(requests_data)}")
            result = await tester.test_single_prediction(request)
            results.append(result)

            # Небольшая пауза между запросами
            await asyncio.sleep(0.1)

    # Анализируем результаты
    logger.info("Analyzing results")
    analysis = tester.analyze_results(results)

    # Выводим отчет
    print("\n" + "=" * 60)
    print("🧪 UCI HYDRAULIC ML API TEST REPORT")
    print("=" * 60)

    print("\n📊 SUMMARY:")
    print(f"  Total Tests: {analysis['total_tests']}")
    print(f"  Success Rate: {analysis['success_rate']:.1f}%")
    print(f"  Failed Tests: {analysis['failed_tests']}")

    if analysis.get("latency_stats"):
        latency = analysis["latency_stats"]
        print("\n⚡ PERFORMANCE:")
        print(f"  Request Time p50: {latency['request_time_p50_ms']:.1f}ms")
        print(f"  Request Time p90: {latency['request_time_p90_ms']:.1f}ms")
        print(f"  Request Time p99: {latency['request_time_p99_ms']:.1f}ms")
        print(f"  API Processing p90: {latency['api_time_p90_ms']:.1f}ms")
        print(f"  Meets Target (<100ms p90): {'✅' if latency['meets_target'] else '❌'}")

    if analysis.get("prediction_stats"):
        pred = analysis["prediction_stats"]
        print("\n🎯 PREDICTIONS:")
        print(f"  Average Ensemble Score: {pred['ensemble_score_mean']:.3f}")
        print(f"  Score Std Deviation: {pred['ensemble_score_std']:.3f}")
        print(f"  Average Confidence: {pred['confidence_mean']:.3f}")
        print("  Severity Distribution:")
        for severity, count in pred["severity_distribution"].items():
            print(f"    {severity}: {count} ({count / analysis['successful_tests'] * 100:.1f}%)")

    if analysis.get("errors"):
        print("\n❌ ERRORS:")
        for error in analysis["errors"][:5]:  # Показываем первые 5
            print(f"  - {error}")

    # Сохраняем полный отчет
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    full_report = {
        "test_config": {
            "api_base": args.api,
            "cycles_tested": args.cycles,
            "data_source": args.data,
            "batch_mode": args.batch,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "analysis": analysis,
        "raw_results": results[:10],  # Первые 10 результатов для детального анализа
    }

    with report_path.open("w") as f:  # Path.open вместо open()
        json.dump(full_report, f, indent=2)

    print(f"\n📄 Full report saved to: {report_path}")
    print("\n" + "=" * 60)

    # Рекомендации
    if analysis.get("latency_stats", {}).get("meets_target"):
        print("✅ PASS: Latency requirements met!")
    else:
        print("⚠️  OPTIMIZATION NEEDED: Consider model optimization or caching")

    if analysis.get("success_rate", 0) > 95:
        print("✅ PASS: High success rate!")
    else:
        print("⚠️  STABILITY ISSUES: Check error patterns")


if __name__ == "__main__":
    asyncio.run(main())
