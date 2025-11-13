#!/usr/bin/env python3
"""
HYDRAULIC SERVICE - Automated Setup (Windows Compatible)
Works on Windows, Linux, Mac - everywhere Python exists!

Usage:
    python setup-production.py

Result: hydraulic-service-production.zip
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# ANSI Colors (disabled on Windows)
class Colors:
    BLUE = ''
    GREEN = ''
    YELLOW = ''
    RED = ''
    END = ''
    BOLD = ''

class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = datetime.now()
        # Clear previous log
        Path(log_file).write_text('')
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def success(self, message: str):
        line = f"[OK] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def error(self, message: str):
        line = f"[ERROR] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def warning(self, message: str):
        line = f"[WARNING] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    
    def header(self, message: str):
        print("")
        print("=" * 60)
        print("  " + message)
        print("=" * 60)
        print("")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n  {message}\n{'='*60}\n\n")

# File definitions
FILES: Dict[str, str] = {
    'services/__init__.py': '',
    'services/api/__init__.py': '',
    'services/api/app/__init__.py': '',
    'services/api/app/routes/__init__.py': '',
    'services/ml/__init__.py': '',
    
    'services/api/app/exceptions.py': '''"""Global exception handling for production-grade API."""
from typing import Any
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    DATABASE_ERROR = "DATABASE_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT = "TIMEOUT"

class AppException(Exception):
    """Base exception for application-level errors."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(message)

class ValidationError(AppException):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=422,
            details=details,
        )

class InferenceError(AppException):
    """Raised when model inference fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.INFERENCE_FAILED,
            message=message,
            status_code=503,
            details=details,
        )

class DatabaseError(AppException):
    """Raised when database operation fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            status_code=503,
            details=details,
        )
''',

    'services/api/app/middleware.py': '''"""Production middleware stack for FastAPI."""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
import uuid
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """Global exception handler with trace_id correlation."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} {response.status_code}",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration,
                }
            )
            return response
            
        except Exception as exc:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {str(exc)}",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration,
                    "exception": type(exc).__name__,
                }
            )
            
            from app.exceptions import AppException, ErrorCode
            
            if isinstance(exc, AppException):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": exc.code.value,
                        "message": exc.message,
                        "trace_id": request_id,
                    },
                )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": ErrorCode.INTERNAL_ERROR.value,
                    "message": "Internal server error",
                    "trace_id": request_id,
                },
            )
''',

    'services/api/app/main.py': '''"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging.config

from app.config import settings

logging.config.dictConfig(settings.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hydraulic Diagnostic Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")

@app.get("/health")
async def health():
    return {"status": "alive"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
''',

    'services/api/config.py': '''"""Configuration management."""
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "Hydraulic Diagnostic Service"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    DATABASE_URL: str = "postgresql://user:pass@db:5432/hydraulic"
    DATABASE_POOL_SIZE: int = 20
    
    LOG_LEVEL: str = "INFO"
    
    LOGGING_CONFIG = {
        "version": 1,
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": "INFO", "handlers": ["default"]},
    }
    
    class Config:
        env_file = ".env"

settings = Settings()
''',

    'services/api/requirements.txt': '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.1
prometheus-client==0.19.0
python-json-logger==2.0.7
slowapi==0.1.9
aioredis==2.0.1
httpx==0.25.2
python-multipart==0.0.6
email-validator==2.1.0
''',

    'services/ml/__init__.py': '',
    
    'services/ml/models.py': '''"""GNN model architecture for hydraulic diagnostics."""
import torch
import torch.nn as nn

class GNNModel(nn.Module):
    """Graph Neural Network for multi-label classification."""
    
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64, num_classes: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
''',

    'services/ml/requirements.txt': '''torch-geometric==2.5.0
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-cluster==1.6.3
pytorch-lightning==2.2.0
wandb==0.16.3
mlflow==2.11.0
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.4.0
pydantic==2.5.3
protobuf==4.25.2
''',

    'Dockerfile': '''FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-runtime as builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential libpq-dev git && rm -rf /var/lib/apt/lists/*

COPY services/ml/requirements.txt ml-requirements.txt
COPY services/api/requirements.txt api-requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels \\
    -r ml-requirements.txt -r api-requirements.txt || true

FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libpq5 curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* || true && \\
    rm -rf /wheels

COPY services/api/app ./app
COPY services/ml ./ml
COPY services/api/config.py ./config.py

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \\
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512 PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
''',

    'k8s/deployment.yaml': '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: hydraulic-service
  namespace: default
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: hydraulic-service
  template:
    metadata:
      labels:
        app: hydraulic-service
    spec:
      containers:
      - name: api
        image: registry.example.com/hydraulic-service:1.0.0
        ports:
        - name: http
          containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
      terminationGracePeriodSeconds: 30

---
apiVersion: v1
kind: Service
metadata:
  name: hydraulic-service
spec:
  type: ClusterIP
  selector:
    app: hydraulic-service
  ports:
  - name: http
    port: 80
    targetPort: 8000

---
apiVersion: autoscaling.k8s.io/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hydraulic-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hydraulic-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
''',

    'database/init.sql': '''CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sensor_data (
    id BIGSERIAL,
    sensor_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    pressure FLOAT8,
    temperature FLOAT8,
    flow_rate FLOAT8,
    viscosity FLOAT8,
    contamination_level FLOAT8,
    device_id TEXT,
    tenant_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('sensor_data', 'timestamp', if_not_exists => TRUE);
ALTER TABLE sensor_data SET (timescaledb.compress = true);
SELECT add_compression_policy('sensor_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('sensor_data', INTERVAL '5 years', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sensor_id_timestamp ON sensor_data (sensor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tenant_timestamp ON sensor_data (tenant_id, timestamp DESC);
''',

    'README.md': '''# Hydraulic Diagnostic Service - Production Ready

## Quick Start

### Local Development
```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r services/api/requirements.txt
cd services/api
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t hydraulic-service:1.0.0 .
docker run --gpus all -p 8000:8000 hydraulic-service:1.0.0
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get deployment hydraulic-service
```

## Features
- PyTorch 2.8.0 with CUDA 12.1
- Global exception handling
- Health checks for K8s
- TimescaleDB hypertables
- Docker multi-stage build
- Kubernetes auto-scaling
''',

    '.gitignore': '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.env
.venv
venv/
.DS_Store
*.log
.idea/
.vscode/
*.swp
*.swo
'''
}

class SetupBuilder:
    def __init__(self, work_dir: str = '.'):
        self.work_dir = work_dir
        self.output_dir = Path(work_dir) / 'hydraulic-service-production'
        self.log_file = Path(work_dir) / 'setup.log'
        self.logger = Logger(str(self.log_file))
        self.created_files: List[str] = []
    
    def run(self):
        """Execute setup."""
        try:
            self.logger.header("HYDRAULIC DIAGNOSTIC SERVICE - PRODUCTION SETUP")
            self.logger.log(f"Working directory: {self.work_dir}")
            self.logger.log(f"Output directory: {self.output_dir}")
            self.logger.log(f"Total files: {len(FILES)}")
            
            # Clean previous
            if self.output_dir.exists():
                self.logger.warning("Removing previous setup...")
                shutil.rmtree(self.output_dir)
            
            # Create files
            self.logger.header("STEP 1: Creating Files")
            self._create_files()
            
            # Verify
            self.logger.header("STEP 2: Verifying Files")
            self._verify_files()
            
            # Create archive
            self.logger.header("STEP 3: Creating ZIP Archive")
            archive_path = self._create_archive()
            
            # Summary
            self.logger.header("[OK] SETUP COMPLETE!")
            self._print_summary(archive_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            import traceback
            self.logger.log(traceback.format_exc())
            return False
    
    def _create_files(self):
        """Create all files."""
        for filepath, content in FILES.items():
            full_path = self.output_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            self.logger.success(f"Created: {filepath}")
            self.created_files.append(filepath)
    
    def _verify_files(self):
        """Verify all files exist."""
        missing = 0
        for filepath in self.created_files:
            if (self.output_dir / filepath).exists():
                self.logger.success(f"OK: {filepath}")
            else:
                self.logger.error(f"MISSING: {filepath}")
                missing += 1
        
        if missing == 0:
            self.logger.success("All files verified!")
        else:
            raise Exception(f"{missing} files are missing!")
    
    def _create_archive(self) -> Path:
        """Create ZIP archive."""
        archive_name = Path(self.work_dir) / 'hydraulic-service-production.zip'
        
        # Remove old archive
        if archive_name.exists():
            archive_name.unlink()
        
        # Create new archive
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.work_dir)
                    zf.write(file_path, arcname)
        
        self.logger.success(f"Archive created: {archive_name.name}")
        return archive_name
    
    def _print_summary(self, archive_path: Path):
        """Print summary."""
        archive_size = archive_path.stat().st_size / (1024 * 1024)
        
        print("")
        print("=" * 60)
        print("[PRODUCTION PACKAGE READY]")
        print("=" * 60)
        print(f"Archive: {archive_path.name}")
        print(f"Location: {archive_path.absolute()}")
        print(f"Size: {archive_size:.2f} MB")
        print(f"Files: {len(self.created_files)}")
        print("")
        print("Next Steps:")
        print("1. Unzip the archive:")
        print(f"   unzip {archive_path.name} -d /path/to/your/project")
        print("2. Copy files to your project:")
        print("   cp -r hydraulic-service-production/* .")
        print("3. Build Docker image:")
        print("   docker build -t hydraulic-service:1.0.0 .")
        print("4. Deploy to Kubernetes:")
        print("   kubectl apply -f k8s/deployment.yaml")
        print("")
        print(f"Log file: {self.log_file}")
        print("")

def main():
    """Main entry point."""
    try:
        builder = SetupBuilder()
        success = builder.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
