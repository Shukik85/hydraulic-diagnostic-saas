"""
Lightweight GNN Model for GTX 1650 SUPER (4GB VRAM)
Memory-efficient architecture with PyTorch 2.3 optimizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class LightweightGNN(nn.Module):
    """
    Memory-efficient GNN optimized for 4GB VRAM
    - Smaller hidden dimensions
    - Fewer layers
    - Fewer attention heads
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        return x


class MemoryEfficientInference:
    """
    Memory-efficient inference engine for 4GB VRAM
    Features:
    - torch.compile with reduce-overhead mode
    - Mixed precision (FP16)
    - CUDA cache management
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        from config import apply_gpu_config, INFERENCE_CONFIG

        apply_gpu_config()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

        # Compile model (PyTorch 2.3)
        if INFERENCE_CONFIG['compile_model']:
            print("Compiling model...")
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=True
            )

        self._warmup()

    def _warmup(self):
        """Warmup with small graphs"""
        dummy_x = torch.randn(5, 10, device=self.device)
        dummy_edge_index = torch.randint(0, 5, (2, 15), device=self.device)

        with torch.inference_mode():
            for _ in range(3):
                _ = self.model(dummy_x, dummy_edge_index)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Memory-efficient prediction with FP16
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # Mixed precision inference
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            scores = self.model(x, edge_index)

        result = scores.cpu()

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def get_memory_stats(self):
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3

        return {
            "allocated_gb": allocated,
            "max_allocated_gb": max_allocated,
            "free_gb": 4.0 - allocated
        }
