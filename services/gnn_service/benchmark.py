"""
Benchmark script for GTX 1650 SUPER
Measures inference performance
"""
import torch
import time
import numpy as np
from models.lightweight_gnn import LightweightGNN, MemoryEfficientInference

def benchmark(engine, num_nodes=20, num_edges=100, iterations=50):
    """Run benchmark"""
    x = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Warmup
    for _ in range(5):
        _ = engine.predict(x, edge_index)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = engine.predict(x, edge_index)
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }

def main():
    print("=" * 70)
    print("GTX 1650 SUPER BENCHMARK")
    print("=" * 70)
    print()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    model = LightweightGNN()
    engine = MemoryEfficientInference(model)

    # Test different graph sizes
    for nodes, edges in [(10, 50), (20, 100), (30, 200)]:
        print(f"Graph: {nodes} nodes, {edges} edges")
        stats = benchmark(engine, nodes, edges)
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95:  {stats['p95']:.2f}ms")
        print()

    # Memory stats
    mem = engine.get_memory_stats()
    print(f"Memory: {mem.get('allocated_gb', 0):.2f} GB / 4.00 GB")
    print()
    print("✅ Benchmark complete!")

if __name__ == "__main__":
    main()
