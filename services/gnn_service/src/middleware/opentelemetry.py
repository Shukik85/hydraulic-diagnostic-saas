"""OpenTelemetry distributed tracing middleware.

Provides:
- Automatic span creation for all requests
- Request/response attribute capture
- Error tracking with exceptions
- OTLP export to collector

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4318)
    OTEL_SERVICE_NAME: Service name (default: gnn-service)
    OTEL_ENABLED: Enable tracing (default: true)

Usage:
    >>> from src.middleware import setup_opentelemetry, OpenTelemetryMiddleware
    >>> # In lifespan startup:
    >>> setup_opentelemetry()
    >>> # Add middleware:
    >>> app.add_middleware(OpenTelemetryMiddleware)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Graceful degradation: OpenTelemetry is optional
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not installed. Install with: pip install opentelemetry-api "
        "opentelemetry-sdk opentelemetry-instrumentation-fastapi "
        "opentelemetry-exporter-otlp-proto-http"
    )


def setup_opentelemetry(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
) -> None:
    """Setup OpenTelemetry tracing.

    Args:
        service_name: Service name for traces (default: from env or 'gnn-service')
        otlp_endpoint: OTLP collector endpoint (default: from env or http://localhost:4318)

    Environment Variables:
        OTEL_ENABLED: Enable tracing (default: true)
        OTEL_SERVICE_NAME: Service name
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint

    Examples:
        >>> # In FastAPI lifespan:
        >>> setup_opentelemetry(
        ...     service_name="gnn-service",
        ...     otlp_endpoint="http://jaeger:4318"
        ... )
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available, skipping setup")
        return

    # Check if enabled
    enabled = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.info("OpenTelemetry disabled via OTEL_ENABLED=false")
        return

    # Get configuration
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "gnn-service")
    otlp_endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
    )

    try:
        # Create resource
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")

        # Add batch span processor
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        logger.info(
            f"✅ OpenTelemetry initialized: {service_name} -> {otlp_endpoint}"
        )

    except Exception as e:
        logger.error(
            f"⚠️ OpenTelemetry setup failed: {e}. Tracing will be disabled.",
            exc_info=True,
        )


class OpenTelemetryMiddleware(BaseHTTPMiddleware):
    """OpenTelemetry middleware for request tracing.

    Creates spans for each request with:
    - HTTP method, path, status code
    - Request duration
    - Query parameters
    - Response size
    - Errors and exceptions

    Examples:
        >>> app.add_middleware(OpenTelemetryMiddleware)
    """

    def __init__(self, app):
        super().__init__(app)
        self.tracer = None
        if OTEL_AVAILABLE:
            try:
                self.tracer = trace.get_tracer(__name__)
            except Exception as e:
                logger.warning(f"Failed to get tracer: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Create span for request."""
        # If tracer not available, pass through
        if self.tracer is None:
            return await call_next(request)

        # Create span
        span_name = f"{request.method} {request.url.path}"

        try:
            with self.tracer.start_as_current_span(span_name) as span:
                # Add request attributes
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.path", request.url.path)

                # Add query params (if any)
                if request.query_params:
                    span.set_attribute(
                        "http.query_params", str(dict(request.query_params))
                    )

                # Add request ID (if present)
                request_id = request.headers.get("X-Request-ID")
                if request_id:
                    span.set_attribute("request.id", request_id)

                # Process request
                start_time = time.time()
                try:
                    response = await call_next(request)
                    duration_ms = (time.time() - start_time) * 1000

                    # Add response attributes
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.duration_ms", duration_ms)

                    # Add response size (if available)
                    if "content-length" in response.headers:
                        span.set_attribute(
                            "http.response_size_bytes",
                            int(response.headers["content-length"]),
                        )

                    return response

                except Exception as e:
                    # Record exception in span
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        except Exception as e:
            # If span creation fails, log and pass through
            logger.error(f"OpenTelemetry span error: {e}", exc_info=True)
            return await call_next(request)
