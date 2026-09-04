"""Unit tests for models: MLP, Random Forest, GCN, GraphSAGE, GAT, GIN."""

from __future__ import annotations

import torch

from spatial_graph_bench.config.model import GNNConfig, MLPConfig, ModelType
from spatial_graph_bench.models.gnn import BenchmarkGNN
from spatial_graph_bench.models.mlp import MLP
from spatial_graph_bench.models.random_forest import RandomForestBaseline


def test_mlp_forward_and_embeddings():
    in_features = 20
    num_classes = 5
    mlp = MLP(in_features, num_classes, MLPConfig(hidden_dims=[32, 16]))

    x = torch.randn(10, in_features)
    logits = mlp(x)
    emb = mlp.get_embeddings(x)

    assert logits.shape == (10, num_classes)
    assert emb.shape == (10, 16)


def test_random_forest():
    rf = RandomForestBaseline()
    import numpy as np

    X = np.random.randn(30, 10)
    y = np.random.randint(0, 3, size=30)
    rf.fit(X, y)

    preds = rf.predict(X)
    probs = rf.predict_proba(X)
    assert len(preds) == 30
    assert probs.shape == (30, 3)


def test_gnn_architectures_forward():
    in_features = 16
    num_classes = 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x = torch.randn(4, in_features)

    for arch in [ModelType.GCN, ModelType.GRAPHSAGE, ModelType.GAT, ModelType.GIN]:
        gnn = BenchmarkGNN(arch, in_features, num_classes, GNNConfig(hidden_dim=32, num_layers=2))
        out = gnn(x, edge_index)
        assert out.shape == (4, num_classes), f"Failed for {arch}"
