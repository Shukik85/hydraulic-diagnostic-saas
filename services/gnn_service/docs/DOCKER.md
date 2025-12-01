# Docker Deployment Guide - GNN Service

## 📦 Overview

Complete Docker setup for GNN Service with **Python 3.14 + PyTorch 2.8 + CUDA 12.8**.

**Available Dockerfiles:**
- `Dockerfile` - Production (multi-stage, optimized)
- `Dockerfile.dev` - Development (hot reload, dev tools)

---

## 🚀 Quick Start

### Development (Hot Reload)

```bash
# Start development environment
docker-compose --profile dev up

# Access:
# - FastAPI: http://localhost:8002
# - TensorBoard: http://localhost:6007
# - API Docs: http://localhost:8002/docs
```

### Production

```bash
# Start production environment
docker-compose --profile prod up -d

# Access:
# - FastAPI: http://localhost:8002
# - Health check: http://localhost:8002/health
```

---

## 🏗️ Build Process

### Production Image

**Multi-stage build for minimal size:**

```bash
# Build production image
docker build -t gnn-service:latest -f Dockerfile .

# Image size: ~4GB (vs ~8GB with devel)
# Startup time: ~5s
# Security: non-root user, minimal attack surface
```

**Stages:**
1. **Base** - CUDA runtime + Python 3.14
2. **Dependencies** - Install PyTorch + packages
3. **Application** - Copy code, create non-root user

### Development Image

```bash
# Build development image
docker build -t gnn-service:dev -f Dockerfile.dev .

# Image size: ~8GB (includes dev tools)
# Features: hot reload, debugging, full CUDA toolkit
```

---

## 🐳 Docker Compose Profiles

### Development Profile (`dev`)

**Services:**
- `gnn-service-dev` - FastAPI with hot reload
- `tensorboard` - Training metrics visualization
- `timescaledb` - PostgreSQL with TimescaleDB extension
- `redis` - Caching layer

**Features:**
- Hot reload on code changes
- Volume mounts for source code
- Full CUDA toolkit for compilation
- TensorBoard on port 6007
- Debug logging

**Usage:**
```bash
# Start all dev services
docker-compose --profile dev up

# Start specific service
docker-compose --profile dev up gnn-service-dev

# View logs
docker-compose --profile dev logs -f gnn-service-dev

# Stop all
docker-compose --profile dev down
```

### Production Profile (`prod`)

**Services:**
- `gnn-service-prod` - Production FastAPI (2 workers)
- `timescaledb` - PostgreSQL database
- `redis` - Caching layer

**Features:**
- Optimized multi-stage build
- Read-only data mounts
- Resource limits (4 CPUs, 16GB RAM)
- Health checks
- Automatic restart

**Usage:**
```bash
# Start production
docker-compose --profile prod up -d

# Check status
docker-compose --profile prod ps

# View logs
docker-compose --profile prod logs -f gnn-service-prod

# Stop
docker-compose --profile prod down
```

---

## 🎮 GPU Configuration

### Prerequisites

**Install NVIDIA Container Toolkit:**

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.8   |
# |-------------------------------+----------------------+----------------------+
```

### GPU Resource Allocation

**Single GPU:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Specific GPU:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # GPU 0
          capabilities: [gpu]
```

**Multiple GPUs:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
          capabilities: [gpu]
```

---

## 📁 Volume Mounts

### Development Volumes

```yaml
volumes:
  # Hot reload - source code (read-write)
  - ./src:/app/src:rw
  - ./api:/app/api:rw
  - ./tests:/app/tests:rw
  
  # Data directories (read-write)
  - ./data:/app/data:rw
  - ./models:/app/models:rw
  - ./logs:/app/logs:rw
  - ./checkpoints:/app/checkpoints:rw
```

### Production Volumes

```yaml
volumes:
  # Read-only for security
  - ./data:/app/data:ro
  - ./models:/app/models:ro
  
  # Write access only where needed
  - ./logs:/app/logs:rw
  - ./checkpoints:/app/checkpoints:rw
```

---

## 🔧 Environment Variables

### Create `.env` file

```bash
# Copy example
cp .env.example .env

# Edit variables
vim .env
```

### Required Variables

```bash
# Environment
ENVIRONMENT=development  # or production
LOG_LEVEL=INFO

# Model paths
MODEL_PATH=/app/models/best.ckpt
METADATA_PATH=/app/data/equipment_metadata.json

# Database
DATABASE_URL=postgresql://postgres:postgres@timescaledb:5432/hydraulic_db

# Redis
REDIS_URL=redis://redis:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8002
API_WORKERS=2
```

---

## 🐛 Troubleshooting

### Issue: GPU not detected

**Check NVIDIA driver:**
```bash
nvidia-smi
# Should show driver version and CUDA version
```

**Check NVIDIA container toolkit:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

**Solution:**
- Install/update NVIDIA drivers
- Install nvidia-container-toolkit
- Restart Docker daemon

### Issue: Out of memory (OOM)

**Check GPU memory:**
```bash
nvidia-smi
# Look at memory usage
```

**Solutions:**
```yaml
# Reduce batch size in .env
BATCH_SIZE=16  # Instead of 32

# Increase resource limits
deploy:
  resources:
    limits:
      memory: 32G  # Instead of 16G
```

### Issue: Slow build

**Use BuildKit:**
```bash
DOCKER_BUILDKIT=1 docker build -t gnn-service:latest .
```

**Layer caching:**
```bash
# Pull previous image for cache
docker pull gnn-service:latest

# Build with cache
docker build --cache-from gnn-service:latest -t gnn-service:latest .
```

### Issue: Port already in use

**Check port usage:**
```bash
sudo lsof -i :8002
```

**Change port in docker-compose.yml:**
```yaml
ports:
  - "8003:8002"  # Host:Container
```

### Issue: Permission denied (volumes)

**Fix ownership:**
```bash
# Match container user (UID 1000)
sudo chown -R 1000:1000 ./data ./models ./logs
```

---

## 📊 Image Size Optimization

### Current Sizes

```
gnn-service:latest (production)  ~4.0 GB
gnn-service:dev (development)    ~8.0 GB
```

### Optimization Tips

**1. Multi-stage builds:**
```dockerfile
# Separate builder and runtime stages
FROM ... AS dependencies
FROM ... AS application
```

**2. Minimize layers:**
```dockerfile
# Combine RUN commands
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*
```

**3. Use .dockerignore:**
```
# Exclude unnecessary files
__pycache__
*.pyc
.git
checkpoints/
logs/
```

**4. Slim base images:**
```dockerfile
# Use runtime instead of devel for production
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu24.04
```

---

## 🔒 Security Best Practices

### 1. Non-root user

```dockerfile
# Create user with UID 1000
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser
```

### 2. Read-only volumes (production)

```yaml
volumes:
  - ./data:/app/data:ro
  - ./models:/app/models:ro
```

### 3. Resource limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
```

### 4. Health checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## 📚 Additional Resources

- **PyTorch Docker:** https://hub.docker.com/r/pytorch/pytorch
- **NVIDIA CUDA:** https://hub.docker.com/r/nvidia/cuda
- **Docker Compose:** https://docs.docker.com/compose/
- **NVIDIA Container Toolkit:** https://github.com/NVIDIA/nvidia-docker

---

## 🎯 Next Steps

1. ✅ **Setup environment** - Install Docker + NVIDIA toolkit
2. ✅ **Create .env file** - Configure variables
3. ✅ **Start development** - `docker-compose --profile dev up`
4. ✅ **Test API** - http://localhost:8002/docs
5. ✅ **Train model** - See docs/TRAINING.md
6. ✅ **Deploy production** - `docker-compose --profile prod up -d`

---

**Happy Containerizing! 🐳**
