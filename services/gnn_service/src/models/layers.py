"""Custom layers для GNN модели.

Специализированные layers:
- EdgeConditionedGATv2Layer - GATv2 с edge features
- ARMAAttentionLSTM - LSTM с ARMA-based attention (ICLR 2025)
- SpectralTemporalLayer - FFT-based temporal processing

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class EdgeConditionedGATv2Layer(nn.Module):
    """GATv2 layer с edge-conditioned attention.
    
    Расширение GATv2 для учёта edge features (diameter, length, material, etc.).
    Важно для hydraulic systems, где свойства соединений влияют на распространение проблем.
    
    GATv2 vs GAT:
        - GAT: attention = LeakyReLU(a^T [Wh_i || Wh_j])
        - GATv2: attention = a^T LeakyReLU(W [h_i || h_j])  # Dynamic!
    
    Edge Conditioning:
        attention = attention * edge_weight_fn(edge_features)
    
    Args:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality (per head)
        heads: Number of attention heads
        dropout: Dropout rate
        edge_dim: Edge feature dimensionality
        concat: Concatenate multi-head outputs (True) or average (False)
        add_self_loops: Add self-loops to graph
        bias: Use bias in linear layers
    
    Examples:
        >>> layer = EdgeConditionedGATv2Layer(
        ...     in_channels=128,
        ...     out_channels=16,  # 128 / 8 heads
        ...     heads=8,
        ...     edge_dim=8
        ... )
        >>> out = layer(x, edge_index, edge_attr)
        >>> out.shape  # [N, 128] (16 * 8 heads)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.3,
        edge_dim: int | None = None,
        concat: bool = True,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.concat = concat

        # Base GATv2 layer
        self.gatv2 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,  # Edge features support
            bias=bias,
            share_weights=False  # Separate weights for source/target
        )

        # Edge feature gating (для управления влиянием edge features)
        if edge_dim is not None:
            self.edge_gate = nn.Sequential(
                nn.Linear(edge_dim, heads),
                nn.Sigmoid()
            )
        else:
            self.edge_gate = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features [N, F_in]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, F_edge]
            return_attention_weights: Return attention weights
        
        Returns:
            out: Updated node features [N, F_out * heads] or [N, F_out]
            attention: (edge_index, attention_weights) if requested
        """
        # GATv2 forward
        if return_attention_weights:
            out, (edge_idx, alpha) = self.gatv2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )

            # Apply edge gating (если есть edge features)
            if self.edge_gate is not None and edge_attr is not None:
                edge_gates = self.edge_gate(edge_attr)  # [E, heads]
                # Modulate attention by edge importance
                alpha = alpha * edge_gates
                # Re-normalize
                alpha = softmax(alpha, edge_idx[1], num_nodes=x.size(0))

            return out, (edge_idx, alpha)
        out = self.gatv2(x, edge_index, edge_attr=edge_attr)
        return out


class ARMAAttentionLSTM(nn.Module):
    """LSTM с ARMA-based attention mechanism.
    
    Autoregressive Moving-Average (ARMA) attention для time series.
    Интегрирует:
    - AR component: исторические тренды
    - MA component: инерционные процессы
    
    Reference:
        Autoregressive Moving-average Attention Mechanism for Time Series
        Forecasting (ICLR 2025 submission)
        Результат: +9.1% improvement в forecasting accuracy
    
    Args:
        input_dim: Input feature dimensionality
        hidden_dim: LSTM hidden state dimensionality
        num_layers: Number of LSTM layers
        ar_order: Autoregressive order (typically 3-5)
        ma_order: Moving average order (typically 2-3)
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
    
    Examples:
        >>> lstm = ARMAAttentionLSTM(
        ...     input_dim=128,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ...     ar_order=3,
        ...     ma_order=2
        ... )
        >>> out, (h_n, c_n) = lstm(x)  # x: [B, T, 128]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        ar_order: int = 3,
        ma_order: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.bidirectional = bidirectional

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # ARMA attention parameters
        # AR: авторегрессионные веса (учёт истории)
        self.ar_weights = nn.Parameter(torch.randn(ar_order))

        # MA: moving average веса (сглаживание)
        self.ma_weights = nn.Parameter(torch.randn(ma_order))

        # Attention query/key projections
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.query_proj = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.key_proj = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.value_proj = nn.Linear(lstm_output_dim, lstm_output_dim)

        # Output projection
        self.output_proj = nn.Linear(lstm_output_dim, lstm_output_dim)

    def _compute_arma_attention(
        self,
        lstm_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute ARMA-based attention weights.
        
        Args:
            lstm_out: LSTM output [B, T, H]
        
        Returns:
            attention: Attention weights [B, T, 1]
        """
        B, T, H = lstm_out.shape

        # Queries, Keys, Values
        Q = self.query_proj(lstm_out)  # [B, T, H]
        K = self.key_proj(lstm_out)  # [B, T, H]

        # Standard attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (H ** 0.5)  # [B, T, T]

        # Apply ARMA modulation
        arma_modulation = self._compute_arma_modulation(T).to(lstm_out.device)  # [T, T]

        # Broadcast ARMA modulation across batch
        attn_scores = attn_scores * arma_modulation.unsqueeze(0)  # [B, T, T]

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, T]

        return attn_weights

    def _compute_arma_modulation(self, seq_len: int) -> torch.Tensor:
        """Compute ARMA modulation matrix.
        
        Args:
            seq_len: Sequence length T
        
        Returns:
            modulation: ARMA modulation matrix [T, T]
        """
        # Create time distance matrix
        time_dists = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        time_dists = time_dists.abs().float()

        # AR component (учёт прошлого)
        ar_component = torch.zeros(seq_len, seq_len)
        for i in range(self.ar_order):
            ar_mask = (time_dists == i + 1).float()
            ar_component += self.ar_weights[i] * ar_mask

        # MA component (сглаживание)
        ma_component = torch.zeros(seq_len, seq_len)
        for i in range(self.ma_order):
            ma_mask = (time_dists <= i + 1).float()
            ma_component += self.ma_weights[i] * ma_mask

        # Combined ARMA modulation
        modulation = torch.exp(ar_component + ma_component)

        return modulation

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input sequence [B, T, F]
            hidden: Initial hidden state (h_0, c_0) or None
        
        Returns:
            output: ARMA-attended output [B, T, H]
            hidden: Final hidden state (h_n, c_n)
        """
        # LSTM processing
        lstm_out, hidden_state = self.lstm(x, hidden)  # [B, T, H]

        # ARMA attention
        attn_weights = self._compute_arma_attention(lstm_out)  # [B, T, T]

        # Apply attention to values
        V = self.value_proj(lstm_out)  # [B, T, H]
        attended = torch.bmm(attn_weights, V)  # [B, T, H]

        # Output projection
        output = self.output_proj(attended)  # [B, T, H]

        # Residual connection
        output = output + lstm_out

        return output, hidden_state


class SpectralTemporalLayer(nn.Module):
    """Spectral-temporal layer для frequency domain processing.
    
    Обработка временных рядов в частотной области.
    Полезно для:
    - Выявление циклических паттернов (pressure oscillations)
    - Обнаружение резонансных частот (cavitation, vibration)
    - Анализ гармоник (harmonics в sensor signals)
    
    Reference:
        Spectral-Temporal GNN (IEEE TPAMI 2025)
        State-of-the-art для long-term forecasting
    
    Args:
        hidden_dim: Hidden dimensionality
        num_frequencies: Number of frequency components to keep
        dropout: Dropout rate
    
    Examples:
        >>> layer = SpectralTemporalLayer(hidden_dim=128, num_frequencies=32)
        >>> out = layer(x)  # x: [B, T, 128]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_frequencies: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies

        # Learnable frequency filters
        self.freq_filter = nn.Parameter(
            torch.ones(num_frequencies, hidden_dim)
        )

        # Projection layers
        self.freq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Time series features [B, T, H]
        
        Returns:
            out: Spectral-temporal features [B, T, H]
        """
        B, T, H = x.shape
        residual = x

        # 1. FFT: time domain -> frequency domain
        x_freq = torch.fft.rfft(x, dim=1)  # [B, T//2+1, H] (complex)

        # 2. Keep only top frequencies
        num_freq = min(self.num_frequencies, x_freq.size(1))
        x_freq = x_freq[:, :num_freq, :]  # [B, num_freq, H]

        # 3. Apply learnable frequency filter
        freq_filter = self.freq_filter[:num_freq, :]  # [num_freq, H]
        x_freq_real = x_freq.real * freq_filter
        x_freq_imag = x_freq.imag * freq_filter
        x_freq_filtered = torch.complex(x_freq_real, x_freq_imag)

        # 4. Pad back to original frequency count
        if num_freq < T // 2 + 1:
            padding = torch.zeros(
                B, T // 2 + 1 - num_freq, H,
                dtype=x_freq_filtered.dtype,
                device=x_freq_filtered.device
            )
            x_freq_filtered = torch.cat([x_freq_filtered, padding], dim=1)

        # 5. IFFT: frequency domain -> time domain
        x_time = torch.fft.irfft(x_freq_filtered, n=T, dim=1)  # [B, T, H]

        # 6. Frequency projection
        x_freq_features = self.freq_proj(x_time)

        # 7. Combine with time domain
        x_time_features = self.time_proj(residual)

        # 8. Fusion
        out = x_freq_features + x_time_features
        out = self.dropout(out)
        out = self.layer_norm(out + residual)  # Residual connection

        return out


class DynamicGraphNorm(nn.Module):
    """Dynamic graph normalization layer.
    
    Адаптивная нормализация для графов с разным количеством nodes.
    
    Args:
        hidden_dim: Feature dimensionality
        eps: Epsilon для numerical stability
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [N, H]
            batch: Batch assignment [N] or None
        
        Returns:
            normalized: Normalized features [N, H]
        """
        if batch is None:
            # Single graph - standard normalization
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
        else:
            # Batch of graphs - per-graph normalization
            mean = global_mean_pool(x, batch)  # [B, H]
            # Broadcast back to nodes
            mean = mean[batch]  # [N, H]

            # Compute per-graph std
            centered = x - mean
            var = global_mean_pool(centered ** 2, batch)  # [B, H]
            std = (var + self.eps).sqrt()
            std = std[batch]  # [N, H]

        # Normalize
        normalized = (x - mean) / (std + self.eps)

        # Affine transformation
        normalized = normalized * self.gamma + self.beta

        return normalized
