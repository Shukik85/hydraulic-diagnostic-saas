"""OpenTelemetry middleware для FastAPI.

Предоставляет:
- Автоматическую инструментацию запросов
- Распространение trace context
- Экспорт в OTLP endpoint

Examples:
    >>> from fastapi import FastAPI
    >>> from src.middleware.opentelemetry import setup_opentelemetry
    >>>
    >>> app = FastAPI()
    >>> setup_opentelemetry(app, service_name="gnn-service")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_opentelemetry(app: FastAPI, service_name: str = "gnn-service") -> None:
    """Setup OpenTelemetry instrumentation.

    Args:
        app: FastAPI application
        service_name: Service name for traces
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Setup resource
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Setup tracer provider
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Setup OTLP exporter
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Instrument FastAPI (import only when needed)
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
            logger.info(f"OpenTelemetry instrumentation enabled for {service_name}")
        except ImportError:
            logger.warning("FastAPIInstrumentor not available, skipping FastAPI instrumentation")

    except ImportError as e:
        logger.warning(f"OpenTelemetry not available: {e}")
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}")
