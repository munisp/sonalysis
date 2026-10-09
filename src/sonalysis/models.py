from __future__ import annotations
import torch
from torch import nn

class TransactionMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class GraphFraudEncoder(nn.Module):
    """Lightweight CPU graph encoder: neighborhood message passing is supplied as aggregate features."""
    def __init__(self, n_features: int):
        super().__init__()
        self.self_proj = nn.Linear(n_features, 32)
        self.neighbor_proj = nn.Linear(2, 32)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        # Feature positions 6 and 7 are graph degree and fraud-neighbor rate.
        # They represent the pre-aggregated neighborhood message for CPU inference.
        h = self.self_proj(x) + self.neighbor_proj(x[:, 6:8])
        return self.classifier(h).squeeze(-1)
