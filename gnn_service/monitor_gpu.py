"""
Real-time GPU memory monitoring for GTX 1650 SUPER.
"""

import threading
import time
from pathlib import Path

import pandas as pd
import torch


class GPUMonitor:
    """Monitor GPU memory usage during training."""

    def __init__(self, interval=2):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def get_memory_info(self):
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            cached = torch.cuda.memory_cached() / 1024**3  # GB
            return {
                "timestamp": time.time(),
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "cached_gb": cached,
                "utilization_percent": (allocated / 4.3) * 100,  # 4.3 GB total
            }
        return {}

    def monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            memory_info = self.get_memory_info()
            if memory_info:
                self.data.append(memory_info)
                print(
                    f"GPU Memory: {memory_info['allocated_gb']:.2f} GB "
                    f"({memory_info['utilization_percent']:.1f}%)"
                )
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("GPU monitoring started...")

    def stop(self):
        """Stop monitoring and save report."""
        self.monitoring = False
        if self.thread:
            self.thread.join()

        # Create report
        if self.data:
            df = pd.DataFrame(self.data)
            report_path = Path("logs/gpu_memory_report.csv")
            df.to_csv(report_path, index=False)
            print(f"GPU memory report saved to {report_path}")

            # Summary statistics
            max_usage = df["allocated_gb"].max()
            avg_usage = df["allocated_gb"].mean()
            print(f"Memory usage - Max: {max_usage:.2f}GB, Avg: {avg_usage:.2f}GB")

            if max_usage > 3.5:  # 80% of 4.3GB
                print("⚠️  High memory usage detected! Consider reducing batch size.")
            elif max_usage > 2.5:
                print("✅ Memory usage within safe limits.")
            else:
                print("🎉 Memory usage optimal - you can increase batch size!")


# Usage example
def monitor_training():
    """Monitor GPU during training."""
    monitor = GPUMonitor()
    monitor.start()

    try:
        # Simulate training
        for _i in range(10):
            # Allocate some memory
            x = torch.randn(1000, 1000).cuda()  # noqa: F841
            time.sleep(1)

    finally:
        monitor.stop()


if __name__ == "__main__":
    monitor_training()
