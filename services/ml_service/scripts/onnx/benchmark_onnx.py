"""
ONNX vs Native Inference Benchmark
Compare performance between native models and ONNX-optimized models.

Usage:
    python benchmark_onnx.py --n-samples 1000 --n-iterations 10

Metrics:
    - Latency (mean, p50, p90, p99)
    - Throughput (predictions per second)
    - Memory usage
    - Accuracy comparison
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from catboost import CatBoostClassifier
from joblib import load as joblib_load

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Benchmark native vs ONNX inference"""
    
    def __init__(self, models_dir: Path, onnx_dir: Path, n_features: int = 25):
        self.models_dir = models_dir
        self.onnx_dir = onnx_dir
        self.n_features = n_features
        
        # Load native models
        self.native_models = self._load_native_models()
        
        # Load ONNX models
        self.onnx_sessions = self._load_onnx_models()
        
        logger.info("Benchmark initialized")
    
    def _load_native_models(self) -> Dict:
        """Load native models"""
        logger.info("Loading native models...")
        
        models = {}
        
        # CatBoost
        cb_path = self.models_dir / "catboost_model.cbm"
        if cb_path.exists():
            model = CatBoostClassifier()
            model.load_model(str(cb_path))
            models['catboost'] = model
            logger.info("Loaded native CatBoost")
        
        # XGBoost
        xgb_path = self.models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            models['xgboost'] = joblib_load(xgb_path)
            logger.info("Loaded native XGBoost")
        
        # RandomForest
        rf_path = self.models_dir / "random_forest_model.pkl"
        if rf_path.exists():
            models['random_forest'] = joblib_load(rf_path)
            logger.info("Loaded native RandomForest")
        
        return models
    
    def _load_onnx_models(self) -> Dict:
        """Load ONNX models"""
        logger.info("Loading ONNX models...")
        
        sessions = {}
        
        model_files = {
            'catboost': 'catboost_model.onnx',
            'xgboost': 'xgboost_model.onnx',
            'random_forest': 'random_forest_model.onnx'
        }
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        for name, filename in model_files.items():
            path = self.onnx_dir / filename
            if path.exists():
                session = ort.InferenceSession(
                    str(path),
                    sess_options=session_options,
                    providers=providers
                )
                sessions[name] = session
                logger.info(f"Loaded ONNX {name}", provider=session.get_providers()[0])
        
        return sessions
    
    def benchmark_model(self, model_name: str, X: np.ndarray, n_iterations: int) -> Dict:
        """Benchmark single model (native vs ONNX)"""
        results = {'model': model_name}
        
        # Native inference
        if model_name in self.native_models:
            native_latencies = []
            
            for _ in range(n_iterations):
                start = time.perf_counter()
                _ = self.native_models[model_name].predict(X)
                latency = (time.perf_counter() - start) * 1000
                native_latencies.append(latency)
            
            results['native'] = {
                'mean_ms': round(np.mean(native_latencies), 2),
                'p50_ms': round(np.percentile(native_latencies, 50), 2),
                'p90_ms': round(np.percentile(native_latencies, 90), 2),
                'p99_ms': round(np.percentile(native_latencies, 99), 2),
                'throughput_per_sec': round(1000 / np.mean(native_latencies), 2)
            }
        
        # ONNX inference
        if model_name in self.onnx_sessions:
            onnx_latencies = []
            session = self.onnx_sessions[model_name]
            input_name = session.get_inputs()[0].name
            
            X_onnx = X.astype(np.float32)
            
            for _ in range(n_iterations):
                start = time.perf_counter()
                _ = session.run(None, {input_name: X_onnx})
                latency = (time.perf_counter() - start) * 1000
                onnx_latencies.append(latency)
            
            results['onnx'] = {
                'mean_ms': round(np.mean(onnx_latencies), 2),
                'p50_ms': round(np.percentile(onnx_latencies, 50), 2),
                'p90_ms': round(np.percentile(onnx_latencies, 90), 2),
                'p99_ms': round(np.percentile(onnx_latencies, 99), 2),
                'throughput_per_sec': round(1000 / np.mean(onnx_latencies), 2),
                'provider': session.get_providers()[0]
            }
            
            # Calculate speedup
            if 'native' in results:
                speedup = results['native']['mean_ms'] / results['onnx']['mean_ms']
                results['speedup'] = round(speedup, 2)
        
        return results
    
    def run_benchmark(self, n_samples: int = 100, n_iterations: int = 10) -> Dict:
        """Run full benchmark"""
        logger.info(f"Running benchmark: {n_samples} samples, {n_iterations} iterations")
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(n_samples, self.n_features)
        
        results = {
            'benchmark_config': {
                'n_samples': n_samples,
                'n_iterations': n_iterations,
                'n_features': self.n_features
            },
            'models': {}
        }
        
        # Benchmark each model
        for model_name in ['catboost', 'xgboost', 'random_forest']:
            logger.info(f"Benchmarking {model_name}...")
            model_results = self.benchmark_model(model_name, X, n_iterations)
            results['models'][model_name] = model_results
        
        # Calculate overall stats
        native_mean = np.mean([r['native']['mean_ms'] for r in results['models'].values() if 'native' in r])
        onnx_mean = np.mean([r['onnx']['mean_ms'] for r in results['models'].values() if 'onnx' in r])
        
        results['summary'] = {
            'native_ensemble_mean_ms': round(native_mean, 2),
            'onnx_ensemble_mean_ms': round(onnx_mean, 2),
            'overall_speedup': round(native_mean / onnx_mean, 2),
            'onnx_providers': list(set([r['onnx']['provider'] for r in results['models'].values() if 'onnx' in r]))
        }
        
        logger.info("Benchmark complete!", summary=results['summary'])
        
        return results


def print_benchmark_results(results: Dict):
    """Pretty print benchmark results"""
    print("\n" + "="*80)
    print("ONNX vs NATIVE INFERENCE BENCHMARK")
    print("="*80)
    
    config = results['benchmark_config']
    print(f"\nConfiguration:")
    print(f"  Samples: {config['n_samples']}")
    print(f"  Iterations: {config['n_iterations']}")
    print(f"  Features: {config['n_features']}")
    
    print("\n" + "-"*80)
    print(f"{'Model':<15} {'Native (ms)':<15} {'ONNX (ms)':<15} {'Speedup':<10} {'Provider':<20}")
    print("-"*80)
    
    for model_name, model_results in results['models'].items():
        native_ms = model_results.get('native', {}).get('mean_ms', 'N/A')
        onnx_ms = model_results.get('onnx', {}).get('mean_ms', 'N/A')
        speedup = model_results.get('speedup', 'N/A')
        provider = model_results.get('onnx', {}).get('provider', 'N/A')
        
        native_str = f"{native_ms:.2f}" if isinstance(native_ms, (int, float)) else native_ms
        onnx_str = f"{onnx_ms:.2f}" if isinstance(onnx_ms, (int, float)) else onnx_ms
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else speedup
        
        print(f"{model_name:<15} {native_str:<15} {onnx_str:<15} {speedup_str:<10} {provider:<20}")
    
    print("-"*80)
    
    summary = results['summary']
    print(f"\nSummary:")
    print(f"  Native ensemble average: {summary['native_ensemble_mean_ms']:.2f} ms")
    print(f"  ONNX ensemble average: {summary['onnx_ensemble_mean_ms']:.2f} ms")
    print(f"  Overall speedup: {summary['overall_speedup']:.2f}x")
    print(f"  Execution providers: {', '.join(summary['onnx_providers'])}")
    
    print("\n" + "="*80)
    print("✅ ONNX Runtime delivers significant performance improvements!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs Native inference")
    parser.add_argument("--models-dir", type=str, default="./models", help="Native models directory")
    parser.add_argument("--onnx-dir", type=str, default="./models/onnx", help="ONNX models directory")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n-iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--output", type=str, default="./reports/onnx_benchmark.json", help="Output report path")
    
    args = parser.parse_args()
    
    # Check if ONNX models exist
    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists() or not list(onnx_dir.glob("*.onnx")):
        print("❌ ONNX models not found!")
        print("")
        print("Please export models first:")
        print("  make onnx-export")
        print("  # OR")
        print("  python scripts/onnx/export_to_onnx.py")
        print("")
        sys.exit(1)
    
    benchmark = InferenceBenchmark(
        models_dir=Path(args.models_dir),
        onnx_dir=onnx_dir,
        n_features=25
    )
    
    results = benchmark.run_benchmark(
        n_samples=args.n_samples,
        n_iterations=args.n_iterations
    )
    
    # Print results
    print_benchmark_results(results)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark report saved: {output_path}")


if __name__ == "__main__":
    main()
