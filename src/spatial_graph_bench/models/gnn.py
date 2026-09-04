"""Co-equal pre-registered GNN architectures: GCN, GraphSAGE, GAT, GIN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from spatial_graph_bench.config.model import GNNConfig, ModelType


class BenchmarkGNN(nn.Module):
    """Unified GNN model implementing the co-equal architecture set."""

    def __init__(
        self,
        model_type: ModelType,
        in_features: int,
        num_classes: int,
        config: GNNConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.config = config or GNNConfig()

        hidden_dim = self.config.hidden_dim
        num_layers = self.config.num_layers
        dropout = self.config.dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if self.config.use_batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() if self.config.activation == "relu" else nn.GELU()

        current_in = in_features
        for i in range(num_layers):
            current_out = hidden_dim

            if model_type == ModelType.GCN:
                conv = GCNConv(current_in, current_out)
            elif model_type == ModelType.GRAPHSAGE:
                conv = SAGEConv(current_in, current_out, aggr=self.config.sage_aggr)
            elif model_type == ModelType.GAT:
                heads = self.config.gat_heads if i < num_layers - 1 else 1
                out_per_head = current_out // heads if i < num_layers - 1 else current_out
                conv = GATConv(current_in, out_per_head, heads=heads, concat=(i < num_layers - 1))
            elif model_type == ModelType.GIN:
                mlp_gin = nn.Sequential(
                    nn.Linear(current_in, current_out),
                    nn.BatchNorm1d(current_out) if self.config.use_batch_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(current_out, current_out),
                )
                conv = GINConv(mlp_gin, eps=self.config.gin_eps, train_eps=True)
            else:
                raise ValueError(f"Unsupported GNN type: {model_type}")

            self.convs.append(conv)
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(current_out))
            current_in = current_out

        self.classifier = nn.Linear(current_in, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            if self.model_type == ModelType.GCN and edge_weight is not None:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)

            if self.bns is not None:
                h = self.bns[i](h)

            h = self.relu(h)
            h = self.dropout(h)

        return self.classifier(h)

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            if self.model_type == ModelType.GCN and edge_weight is not None:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)

            if self.bns is not None:
                h = self.bns[i](h)
            h = self.relu(h)
        return h
