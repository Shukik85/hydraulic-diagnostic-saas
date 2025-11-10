"""
Training script with memory optimizations for GTX 1650 SUPER.
"""

import argparse
import logging

from config import model_config, training_config
from memory_optimizations import (
    MemoryOptimizer,
    apply_memory_optimizations,
    clear_gpu_cache,
)
from train import HydraulicTrainer

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# ... остальной код


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gtx1650_config():
    """Apply optimizations for GTX 1650 SUPER."""

    # Memory optimizations
    training_config.batch_size = 12
    training_config.gradient_accumulation_steps = 4
    training_config.num_workers = 2

    # Model size optimizations
    model_config.hidden_dim = 96
    model_config.num_gat_layers = 2
    model_config.num_lstm_layers = 1
    model_config.sequence_length = 5

    # GPU specific
    training_config.gpu_config.memory_fraction = 0.7
    training_config.gpu_config.torch_compile = False
    training_config.gpu_config.gradient_checkpointing = True

    logger.info("GTX 1650 SUPER configuration applied")


class MemoryOptimizedTrainer(HydraulicTrainer):
    """Trainer with additional memory optimizations."""

    def setup_model(self):
        """Initialize model with memory optimizations."""
        model = super().setup_model()

        # Apply memory optimizations
        apply_memory_optimizations(model, self.config)
        logger.info("Memory optimizations applied to model")

        return model

    def train_epoch(self, train_loader):
        """Train epoch with memory monitoring."""
        # Use memory optimizer context manager
        with MemoryOptimizer() as mem_opt:
            mem_opt.checkpoint("start_of_epoch")

            result = super().train_epoch(train_loader)

            mem_opt.checkpoint("end_of_epoch")
            return result

    def validate_epoch(self, val_loader):
        """Validate epoch with memory cleanup."""
        with MemoryOptimizer():
            return super().validate_epoch(val_loader)


def main():
    """Main training function with memory optimizations."""

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--monitor", action="store_true", help="Enable memory monitoring"
    )
    args = parser.parse_args()

    # Apply GTX 1650 optimizations
    setup_gtx1650_config()

    # Update from command line
    training_config.epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.gradient_accumulation_steps = args.grad_accum

    # Clear memory before starting
    clear_gpu_cache()

    try:
        # Start training with memory optimizations
        trainer = MemoryOptimizedTrainer(training_config)

        # Enable memory monitoring if requested
        if args.monitor:
            logger.info("Memory monitoring enabled")
            # Здесь можно добавить вызов monitor_gpu.py

        trainer.train()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU Out of Memory! Try these fixes:")
            logger.error("1. Reduce batch_size: --batch_size 8")
            logger.error("2. Increase gradient accumulation: --grad_accum 8")
            logger.error("3. Use --monitor to track memory usage")
        raise


if __name__ == "__main__":
    main()
