"""Tuned spatially ignorant MLP baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from spatial_graph_bench.config.model import MLPConfig


class MLP(nn.Module):
    """Multi-Layer Perceptron tabular baseline ignoring spatial coordinates."""

    def __init__(self, in_features: int, num_classes: int, config: MLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPConfig()

        layers: list[nn.Module] = []
        current_dim = in_features

        for h_dim in self.config.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU() if self.config.activation == "relu" else nn.GELU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
