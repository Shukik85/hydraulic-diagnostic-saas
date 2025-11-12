"""
Temporal RUL (Remaining Useful Life) Prediction Model
Предсказание времени до отказа компонентов гидравлической системы

Architecture:
- Temporal GAT: Пространственные зависимости между компонентами
- Bi-LSTM: Временная динамика деградации (прошлое + будущее)
- Multi-Horizon Heads: Предсказание RUL на 5, 15, 30 минут
- Uncertainty Estimation: Confidence scores

Optimizations:
- PyTorch 2.5.1 torch.compile
- Mixed Precision (FP16)
- Gradient Checkpointing
- Dynamic batch sizes

References:
- [52] Short-Horizon Predictive Maintenance (2025)
- [56] Predictive Maintenance using LSTM and Adaptive Windowing (2024)
- [57] ML-Based RUL Predictions (2024)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv, global_mean_pool

logger = logging.getLogger(__name__)

# Enable optimizations
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


class TemporalGATBlock(nn.Module):
    """Single timestep GAT processing block."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * num_heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process single timestep."""
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class RULPredictionHead(nn.Module):
    """Multi-task RUL prediction head with uncertainty."""
    
    def __init__(self, input_dim: int, num_components: int):
        super().__init__()
        
        # RUL regression (0-1, где 0 = imminent failure, 1 = healthy)
        self.rul_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(input_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_components),
            nn.Sigmoid()  # 0-1 range
        )
        
        # Uncertainty estimation (алеаторическая + эпистемическая)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_components),
            nn.Softplus()  # Always positive
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (rul_scores, uncertainty_scores)."""
        rul = self.rul_head(x)
        uncertainty = self.uncertainty_head(x)
        return rul, uncertainty


class TemporalRULPredictor(nn.Module):
    """
    Temporal GNN для предсказания RUL (Remaining Useful Life).
    
    Input:
        - x_sequence: [batch, time_steps, n_nodes, n_features]
        - edge_index: [2, n_edges]
        - batch: [batch_size * n_nodes]
    
    Output:
        - rul_predictions: Dict[horizon, [batch, n_nodes]]
        - confidence_scores: Dict[horizon, [batch, n_nodes]]
    
    Horizons:
        - 5 minutes: Emergency stop decision
        - 15 minutes: Planned shutdown
        - 30 minutes: Maintenance scheduling
    """
    
    def __init__(
        self,
        num_node_features: int = 15,
        hidden_dim: int = 96,
        num_components: int = 7,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        prediction_horizons: List[int] = [5, 15, 30],  # minutes
        use_checkpointing: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.num_gat_layers = num_gat_layers
        self.prediction_horizons = prediction_horizons
        self.use_checkpointing = use_checkpointing
        
        # GAT layers для пространственной агрегации
        self.gat_blocks = nn.ModuleList()
        for i in range(num_gat_layers):
            in_ch = num_node_features if i == 0 else hidden_dim * num_heads
            self.gat_blocks.append(
                TemporalGATBlock(
                    in_channels=in_ch,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
        
        # Bi-LSTM для временной динамики
        lstm_input_size = hidden_dim * num_heads
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,  # ✅ Bi-directional для лучшего контекста
        )
        
        # Multi-horizon RUL prediction heads
        self.rul_heads = nn.ModuleDict({
            f"horizon_{h}min": RULPredictionHead(
                input_dim=hidden_dim * 2,  # *2 for bidirectional
                num_components=num_components,
            )
            for h in prediction_horizons
        })
        
        logger.info(
            f"TemporalRULPredictor initialized: "
            f"{num_gat_layers} GAT layers, {num_lstm_layers} LSTM layers, "
            f"horizons={prediction_horizons} min"
        )
    
    def _process_timestep(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Process single timestep through GAT (for checkpointing)."""
        return self.gat_blocks[layer_idx](x, edge_index)
    
    def forward(
        self,
        x_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with temporal modeling.
        
        Args:
            x_sequence: [batch, time_steps, n_nodes, n_features]
            edge_index: [2, n_edges]
            batch: [batch_size * n_nodes] (optional)
        
        Returns:
            rul_predictions: {"horizon_5min": [batch, n_nodes], ...}
            confidence_scores: {"horizon_5min": [batch, n_nodes], ...}
        """
        batch_size, T, N, F = x_sequence.shape
        device = x_sequence.device
        
        # Process each timestep через GAT layers
        temporal_embeddings = []
        
        for t in range(T):
            # Reshape для graph processing: [batch*n_nodes, features]
            x_t = x_sequence[:, t].reshape(batch_size * N, F)
            
            # GAT layers
            for layer_idx in range(self.num_gat_layers):
                if self.use_checkpointing and self.training:
                    x_t = checkpoint(
                        self._process_timestep,
                        x_t,
                        edge_index,
                        layer_idx,
                        use_reentrant=False,
                    )
                else:
                    x_t = self._process_timestep(x_t, edge_index, layer_idx)
            
            # Reshape back: [batch, n_nodes, hidden]
            x_t = x_t.view(batch_size, N, -1)
            temporal_embeddings.append(x_t)
        
        # Stack temporal embeddings: [batch, time_steps, n_nodes, hidden]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        
        # Process each node's temporal sequence через LSTM
        rul_predictions = {}
        confidence_scores = {}
        
        for node_idx in range(N):
            # Extract node sequence: [batch, time_steps, hidden]
            node_sequence = temporal_embeddings[:, :, node_idx, :]
            
            # LSTM processing
            lstm_out, (h_n, c_n) = self.lstm(node_sequence)
            
            # Use final state from both directions: [batch, hidden*2]
            final_state = lstm_out[:, -1, :]
            
            # Predict RUL for each horizon
            for horizon_name, rul_head in self.rul_heads.items():
                rul, uncertainty = rul_head(final_state)
                
                # Store predictions
                if horizon_name not in rul_predictions:
                    rul_predictions[horizon_name] = []
                    confidence_scores[horizon_name] = []
                
                rul_predictions[horizon_name].append(rul[:, node_idx:node_idx+1])
                confidence_scores[horizon_name].append(uncertainty[:, node_idx:node_idx+1])
        
        # Stack predictions: [batch, n_nodes]
        rul_predictions = {
            k: torch.cat(v, dim=1)
            for k, v in rul_predictions.items()
        }
        confidence_scores = {
            k: torch.cat(v, dim=1)
            for k, v in confidence_scores.items()
        }
        
        return rul_predictions, confidence_scores


def create_temporal_rul_model(
    num_node_features: int = 15,
    num_components: int = 7,
    device: str = "cuda",
    use_compile: bool = True,
    compile_mode: str = "reduce-overhead",
) -> TemporalRULPredictor:
    """
    Create and compile Temporal RUL model.
    
    Args:
        num_node_features: Number of features per node
        num_components: Number of hydraulic components
        device: 'cuda' or 'cpu'
        use_compile: Enable torch.compile
        compile_mode: 'default', 'reduce-overhead', or 'max-autotune'
    
    Returns:
        Compiled model ready for training/inference
    """
    model = TemporalRULPredictor(
        num_node_features=num_node_features,
        num_components=num_components,
        hidden_dim=96,
        num_gat_layers=3,
        num_heads=4,
        num_lstm_layers=2,
        prediction_horizons=[5, 15, 30],
    ).to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, GATv2Conv):
            if hasattr(m, "lin_l") and m.lin_l is not None:
                nn.init.xavier_uniform_(m.lin_l.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
    
    model.apply(init_weights)
    
    # Compile model (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        logger.info(f"Compiling Temporal RUL model with mode={compile_mode}...")
        model = torch.compile(
            model,
            mode=compile_mode,
            dynamic=True,
        )
        logger.info("✅ Model compiled!")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Temporal RUL model created on {device}")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model


class RULLoss(nn.Module):
    """Custom loss for RUL prediction with uncertainty."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Args:
            alpha: Weight for RUL MSE loss
            beta: Weight for uncertainty regularization
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self,
        rul_pred: torch.Tensor,
        rul_target: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative Log-Likelihood with uncertainty.
        
        Args:
            rul_pred: [batch, n_nodes]
            rul_target: [batch, n_nodes]
            uncertainty: [batch, n_nodes]
        """
        # MSE loss weighted by uncertainty
        mse_loss = F.mse_loss(rul_pred, rul_target, reduction="none")
        
        # Negative log-likelihood with Gaussian assumption
        nll_loss = (
            0.5 * torch.log(uncertainty + 1e-6) +
            0.5 * mse_loss / (uncertainty + 1e-6)
        )
        
        # Regularization to prevent uncertainty collapse
        uncertainty_reg = -torch.log(uncertainty + 1e-6)
        
        total_loss = (
            self.alpha * nll_loss.mean() +
            self.beta * uncertainty_reg.mean()
        )
        
        return total_loss


if __name__ == "__main__":
    # Test temporal RUL model
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Testing Temporal RUL model on {device}")
    
    # Create model
    model = create_temporal_rul_model(
        num_node_features=15,
        num_components=7,
        device=device,
        use_compile=True,
    )
    
    # Dummy temporal data
    batch_size = 2
    time_steps = 12  # 12 × 5min = 60min lookback
    n_nodes = 7
    n_features = 15
    
    # [batch, time_steps, n_nodes, n_features]
    x_sequence = torch.randn(batch_size, time_steps, n_nodes, n_features).to(device)
    
    # Star topology
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
         [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            rul_preds, confidences = model(x_sequence, edge_index)
    
    # Print results
    print("\n" + "=" * 70)
    print("TEMPORAL RUL MODEL TEST")
    print("=" * 70)
    
    for horizon_name, rul in rul_preds.items():
        confidence = confidences[horizon_name]
        print(f"\n{horizon_name}:")
        print(f"  RUL shape: {rul.shape}")
        print(f"  RUL range: [{rul.min():.3f}, {rul.max():.3f}]")
        print(f"  Confidence shape: {confidence.shape}")
        print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("✅ Temporal RUL model test passed!")
    print("=" * 70)
