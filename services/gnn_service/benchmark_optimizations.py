#!/usr/bin/env python
"""
Benchmark PyTorch 2.5.1 Optimizations

Tests:
1. Baseline (PyTorch 2.5, no optimizations)
2. torch.compile (mode=default)
3. torch.compile (mode=reduce-overhead)
4. Mixed precision (FP16)
5. Combined (compile + FP16)

Metrics:
- Latency (p50, p95, p99)
- Throughput (graphs/sec)
- Memory usage (peak VRAM)
- Compilation time
"""

import gc
import logging
import statistics
import time
from typing import List, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from model_v2 import EnhancedTemporalGAT
from model_v2_optimized import EnhancedTemporalGATOptimized, create_optimized_model
from config import model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Comprehensive benchmark suite for PyTorch 2.5.1 optimizations."""

    def __init__(self, device: str = "cuda", num_warmup: int = 10, num_trials: int = 100):
        self.device = device
        self.num_warmup = num_warmup
        self.num_trials = num_trials
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        logger.info(f"Benchmark Suite: device={self.device}, warmup={num_warmup}, trials={num_trials}")

    def create_test_data(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create test data for benchmarking."""
        num_nodes = 7
        num_features = 15
        
        x = torch.randn(batch_size * num_nodes, num_features).to(self.device)
        
        # Star graph topology (pump at center)
        edge_list = []
        for i in range(batch_size):
            offset = i * num_nodes
            for target in range(1, 7):
                edge_list.extend([
                    [offset, offset + target],
                    [offset + target, offset]
                ])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes).to(self.device)
        
        return {
            "x": x,
            "edge_index": edge_index,
            "batch": batch,
        }

    def measure_latency(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor],
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """Measure inference latency."""
        model.eval()
        latencies = []
        
        # Warmup
        for _ in range(self.num_warmup):
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = model(data["x"], data["edge_index"], data["batch"])
                else:
                    _ = model(data["x"], data["edge_index"], data["batch"])
        
        # Synchronize for accurate timing
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        for _ in range(self.num_trials):
            start = time.perf_counter()
            
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits, _, _ = model(data["x"], data["edge_index"], data["batch"])
                else:
                    logits, _, _ = model(data["x"], data["edge_index"], data["batch"])
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
        }

    def measure_memory(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Measure peak memory usage."""
        if self.device != "cuda":
            return {"peak_mb": 0.0, "allocated_mb": 0.0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        model.eval()
        with torch.no_grad():
            _ = model(data["x"], data["edge_index"], data["batch"])
        
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            "peak_mb": peak_memory,
            "allocated_mb": allocated_memory,
        }

    def benchmark_baseline(self) -> Dict[str, Any]:
        """Benchmark baseline model (no optimizations)."""
        logger.info("=" * 70)
        logger.info("Benchmark 1: Baseline (PyTorch 2.5, no optimizations)")
        logger.info("=" * 70)
        
        model = EnhancedTemporalGAT(
            num_node_features=model_config.num_node_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gat_layers=3,
            num_heads=model_config.num_heads,
            gat_dropout=model_config.gat_dropout,
            num_lstm_layers=model_config.num_lstm_layers,
            lstm_dropout=model_config.lstm_dropout,
        ).to(self.device)
        
        data = self.create_test_data(batch_size=1)
        
        latency = self.measure_latency(model, data, use_amp=False)
        memory = self.measure_memory(model, data)
        
        results = {
            "name": "Baseline",
            "latency": latency,
            "memory": memory,
            "throughput": 1000 / latency["p50"],  # graphs/sec
        }
        
        logger.info(f"  Latency p50: {latency['p50']:.2f}ms")
        logger.info(f"  Latency p95: {latency['p95']:.2f}ms")
        logger.info(f"  Throughput: {results['throughput']:.1f} graphs/sec")
        logger.info(f"  Peak memory: {memory['peak_mb']:.1f}MB")
        
        return results

    def benchmark_compiled(
        self,
        compile_mode: str = "reduce-overhead",
    ) -> Dict[str, Any]:
        """Benchmark torch.compile."""
        logger.info("=" * 70)
        logger.info(f"Benchmark 2: torch.compile (mode={compile_mode})")
        logger.info("=" * 70)
        
        model = create_optimized_model(
            device=self.device,
            use_compile=True,
            compile_mode=compile_mode,
        )
        
        data = self.create_test_data(batch_size=1)
        
        # Measure compilation time
        compile_start = time.time()
        with torch.no_grad():
            _ = model(data["x"], data["edge_index"], data["batch"])
        compile_time = time.time() - compile_start
        
        latency = self.measure_latency(model, data, use_amp=False)
        memory = self.measure_memory(model, data)
        
        results = {
            "name": f"torch.compile ({compile_mode})",
            "latency": latency,
            "memory": memory,
            "throughput": 1000 / latency["p50"],
            "compile_time": compile_time,
        }
        
        logger.info(f"  Compilation time: {compile_time:.2f}s")
        logger.info(f"  Latency p50: {latency['p50']:.2f}ms")
        logger.info(f"  Latency p95: {latency['p95']:.2f}ms")
        logger.info(f"  Throughput: {results['throughput']:.1f} graphs/sec")
        logger.info(f"  Peak memory: {memory['peak_mb']:.1f}MB")
        
        return results

    def benchmark_mixed_precision(self) -> Dict[str, Any]:
        """Benchmark mixed precision (FP16)."""
        logger.info("=" * 70)
        logger.info("Benchmark 3: Mixed Precision (FP16)")
        logger.info("=" * 70)
        
        model = EnhancedTemporalGATOptimized(
            num_node_features=model_config.num_node_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gat_layers=3,
            num_heads=model_config.num_heads,
            gat_dropout=model_config.gat_dropout,
            num_lstm_layers=model_config.num_lstm_layers,
            lstm_dropout=model_config.lstm_dropout,
            use_compile=False,
        ).to(self.device)
        
        data = self.create_test_data(batch_size=1)
        
        latency = self.measure_latency(model, data, use_amp=True)
        memory = self.measure_memory(model, data)
        
        results = {
            "name": "Mixed Precision (FP16)",
            "latency": latency,
            "memory": memory,
            "throughput": 1000 / latency["p50"],
        }
        
        logger.info(f"  Latency p50: {latency['p50']:.2f}ms")
        logger.info(f"  Latency p95: {latency['p95']:.2f}ms")
        logger.info(f"  Throughput: {results['throughput']:.1f} graphs/sec")
        logger.info(f"  Peak memory: {memory['peak_mb']:.1f}MB")
        
        return results

    def benchmark_full_optimizations(self) -> Dict[str, Any]:
        """Benchmark all optimizations combined."""
        logger.info("=" * 70)
        logger.info("Benchmark 4: Full Optimizations (compile + FP16)")
        logger.info("=" * 70)
        
        model = create_optimized_model(
            device=self.device,
            use_compile=True,
            compile_mode="reduce-overhead",
        )
        
        data = self.create_test_data(batch_size=1)
        
        # Warmup compilation
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(data["x"], data["edge_index"], data["batch"])
        
        latency = self.measure_latency(model, data, use_amp=True)
        memory = self.measure_memory(model, data)
        
        results = {
            "name": "Full Optimizations",
            "latency": latency,
            "memory": memory,
            "throughput": 1000 / latency["p50"],
        }
        
        logger.info(f"  Latency p50: {latency['p50']:.2f}ms")
        logger.info(f"  Latency p95: {latency['p95']:.2f}ms")
        logger.info(f"  Throughput: {results['throughput']:.1f} graphs/sec")
        logger.info(f"  Peak memory: {memory['peak_mb']:.1f}MB")
        
        return results

    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all benchmarks and generate report."""
        results = []
        
        # 1. Baseline
        results.append(self.benchmark_baseline())
        
        # 2. torch.compile (default)
        if hasattr(torch, "compile"):
            results.append(self.benchmark_compiled(compile_mode="default"))
        
        # 3. torch.compile (reduce-overhead)
        if hasattr(torch, "compile"):
            results.append(self.benchmark_compiled(compile_mode="reduce-overhead"))
        
        # 4. Mixed precision
        if self.device == "cuda":
            results.append(self.benchmark_mixed_precision())
        
        # 5. Full optimizations
        if hasattr(torch, "compile") and self.device == "cuda":
            results.append(self.benchmark_full_optimizations())
        
        self.print_comparison_table(results)
        
        return results

    def print_comparison_table(self, results: List[Dict[str, Any]]):
        """Print comparison table."""
        logger.info("\n" + "=" * 100)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 100)
        
        baseline_p50 = results[0]["latency"]["p50"]
        baseline_memory = results[0]["memory"]["peak_mb"]
        
        print(f"\n{'Configuration':<30} {'p50 (ms)':<12} {'p95 (ms)':<12} {'Speedup':<10} {'Memory (MB)':<12} {'Memory Reduction'}")
        print("-" * 100)
        
        for r in results:
            p50 = r["latency"]["p50"]
            p95 = r["latency"]["p95"]
            speedup = baseline_p50 / p50
            memory = r["memory"]["peak_mb"]
            memory_reduction = (1 - memory / baseline_memory) * 100 if baseline_memory > 0 else 0
            
            print(
                f"{r['name']:<30} "
                f"{p50:>10.2f}  "
                f"{p95:>10.2f}  "
                f"{speedup:>8.2f}x  "
                f"{memory:>10.1f}  "
                f"{memory_reduction:>+8.1f}%"
            )
        
        print("\n" + "=" * 100)
        
        # Best configuration
        best = min(results, key=lambda x: x["latency"]["p50"])
        logger.info(f"\n✅ Best configuration: {best['name']}")
        logger.info(f"   Latency: {best['latency']['p50']:.2f}ms (p50)")
        logger.info(f"   Speedup: {baseline_p50 / best['latency']['p50']:.2f}x")
        logger.info(f"   Memory: {best['memory']['peak_mb']:.1f}MB")
        logger.info(f"   Throughput: {best['throughput']:.1f} graphs/sec")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    suite = BenchmarkSuite(device=device, num_warmup=10, num_trials=100)
    results = suite.run_all_benchmarks()
    
    logger.info("\n✅ Benchmarking completed!")
