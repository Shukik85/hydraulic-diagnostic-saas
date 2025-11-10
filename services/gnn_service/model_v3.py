"""
Model V3: Enhanced GNN for Hydraulic Diagnostics (F1 > 90%)
- Tune dropout schedule
- Fine-tuned LayerNorm and Attention pooling
- Improved initialization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from config_v3 import model_config_v3

class AttentionPoolingV3(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )
    def forward(self, x, batch):
        att_scores = self.attention_net(x)
        att_weights = torch.zeros_like(att_scores)
        for batch_id in torch.unique(batch):
            mask = batch == batch_id
            softmaxed = F.softmax(att_scores[mask], dim=0)
            att_weights[mask] = softmaxed
        return global_mean_pool(x * att_weights, batch)

class ResidualBlockV3(nn.Module):
    def __init__(self, dim, dropout=0.13):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + identity
        out = F.relu(out)
        return out

class EnhancedHydraulicGNNv3(nn.Module):
    def __init__(self):
        super().__init__()
        p = model_config_v3
        # Dropout schedule tuned for robust learning
        self.dropout_schedule = [0.18, 0.13, 0.09][:p.num_gat_layers]
        self.gat_layers = nn.ModuleList()
        for i in range(p.num_gat_layers):
            in_ch = p.num_node_features if i == 0 else p.hidden_dim * p.num_heads
            self.gat_layers.append(
                GATv2Conv(in_channels=in_ch,
                          out_channels=p.hidden_dim,
                          heads=p.num_heads,
                          dropout=p.gat_dropout,
                          concat=True)
            )
        self.ln = nn.ModuleList([
            nn.LayerNorm(p.hidden_dim * p.num_heads) for _ in range(p.num_gat_layers)
        ])
        self.attention_pool = AttentionPoolingV3(p.hidden_dim * p.num_heads)
        self.lstm = nn.LSTM(input_size=p.hidden_dim * p.num_heads,
                            hidden_size=p.hidden_dim,
                            num_layers=p.num_lstm_layers,
                            dropout=p.lstm_dropout if p.num_lstm_layers > 1 else 0,
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(p.hidden_dim, p.hidden_dim),
            nn.LayerNorm(p.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            ResidualBlockV3(p.hidden_dim, dropout=0.13),
            nn.Linear(p.hidden_dim, p.hidden_dim // 2),
            nn.LayerNorm(p.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(p.hidden_dim // 2, p.num_classes),
        )
    def forward(self, x, edge_index, batch=None):
        for i, (gat, ln, drop) in enumerate(zip(self.gat_layers, self.ln, self.dropout_schedule)):
            x, _ = gat(x, edge_index, return_attention_weights=True)
            x = ln(x)
            x = F.elu(x)
            x = F.dropout(x, p=drop, training=self.training)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g_emb = self.attention_pool(x, batch)
        g_emb = g_emb.unsqueeze(1)
        lstm_out, (h_n, _) = self.lstm(g_emb)
        temporal_feat = h_n[-1]
        logits = self.classifier(temporal_feat)
        return logits

def create_enhanced_model_v3(device: str = "cuda"):
    model = EnhancedHydraulicGNNv3().to(device)
    def init_weights(m):
        if isinstance(m, GATv2Conv):
            if hasattr(m, "lin_l") and m.lin_l is not None:
                nn.init.xavier_uniform_(m.lin_l.weight)
                if m.lin_l.bias is not None:
                    nn.init.zeros_(m.lin_l.bias)
            if hasattr(m, "lin_r") and m.lin_r is not None:
                nn.init.xavier_uniform_(m.lin_r.weight)
                if m.lin_r.bias is not None:
                    nn.init.zeros_(m.lin_r.bias)
            if hasattr(m, "att") and m.att is not None:
                nn.init.xavier_uniform_(m.att)
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
    return model
