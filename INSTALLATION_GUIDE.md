# 🚀 Installation Guide for GTX 1650 SUPER

## Your System

- GPU: NVIDIA GeForce GTX 1650 SUPER
- VRAM: 4 GB GDDR6
- Driver: 577.00 (CUDA 12.9)
- Compute: 7.5

## Prerequisites

### 1. NVIDIA Driver (✅ Already installed)
```bash
nvidia-smi
# Should show Driver 577.00, CUDA 12.9
```

### 2. Docker with GPU Support

**Windows WSL2:**
```powershell
# WSL2 должен быть установлен
wsl --version

# Docker Desktop with GPU support
# Download from docker.com
# GPU support enabled by default in Docker Desktop 4.19+
```

**Verify GPU in Docker:**
```bash
wsl
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Installation Steps

### Step 1: Extract Patch

```bash
cd /h/hydraulic-diagnostic-saas
unzip gtx1650_consolidated_patch.zip
```

### Step 2: Apply Files

```bash
# Copy all files
cp -r services/ /h/hydraulic-diagnostic-saas/

# Update docker-compose.yml
cat docker-compose.gnn.patch >> docker-compose.yml
```

### Step 3: Build Image

```bash
cd /h/hydraulic-diagnostic-saas

# Stop old service if running
docker-compose stop gnn_service

# Build (10-15 minutes)
docker-compose build --no-cache gnn_service
```

### Step 4: Start Service

```bash
docker-compose up -d gnn_service

# Wait for startup
sleep 30

# Check logs
docker-compose logs gnn_service
```

### Step 5: Verify Installation

```bash
# Health check
curl http://localhost:8001/health/

# Expected response:
# {
#   "status": "ok",
#   "pytorch_version": "2.3.1",
#   "cuda_available": true,
#   "gpu_name": "NVIDIA GeForce GTX 1650 SUPER"
# }

# Verify GPU inside container
docker-compose exec gnn_service nvidia-smi

# Run verification script
docker-compose exec gnn_service python verify_gpu.py
```

### Step 6: Run Benchmark

```bash
docker-compose exec gnn_service python benchmark.py

# Expected output:
# Graph: 10 nodes, 50 edges
#   Mean: 20-30ms
#   P95:  30-40ms
# 
# Graph: 20 nodes, 100 edges
#   Mean: 40-60ms
#   P95:  60-80ms
#
# Memory: 0.8 GB / 4.0 GB
```

### Step 7: Test Inference

```bash
curl -X POST http://localhost:8001/gnn/infer \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "system_id": "test_001",
    "node_features": [[1,2,3,4,5,6,7,8,9,10]],
    "edge_index": [[0], [0]]
  }'

# Expected inference_time_ms: 20-40ms
```

## Troubleshooting

### Issue 1: GPU not detected

```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall toolkit:
wsl
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue 2: Out of memory

```bash
# Reduce batch size in config.py:
max_batch_size: 3  # Instead of 5

# Or reduce model size:
hidden_channels: 16  # Instead of 32
```

### Issue 3: Slow inference

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check if GPU is actually being used
docker-compose logs gnn_service | grep CUDA
```

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Single inference | 20-40ms |
| Batch (5 graphs) | 100-150ms |
| Memory usage | 0.8-3.5 GB |
| GPU utilization | 60-90% |

## Next Steps

1. ✅ Verify all tests pass
2. ✅ Integrate with backend API
3. ✅ Test with real hydraulic data
4. ✅ Monitor performance in production

---

**Installation complete! Your GTX 1650 SUPER is ready! 🚀**
