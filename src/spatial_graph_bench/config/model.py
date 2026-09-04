"""Model architecture and training hyperparameter configuration schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from spatial_graph_bench.config.base import BaseBenchConfig


class ModelType(StrEnum):
    """Supported model architectures in the benchmark."""

    # Spatially ignorant baselines
    MLP = "mlp"
    RANDOM_FOREST = "random_forest"

    # Pre-registered co-equal GNN architecture set
    GCN = "gcn"
    GRAPHSAGE = "graphsage"
    GAT = "gat"
    GIN = "gin"


class MLPConfig(BaseBenchConfig):
    """Configuration for tuned spatially ignorant MLP baseline."""

    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)
    use_batch_norm: bool = True
    activation: str = "relu"


class RandomForestConfig(BaseBenchConfig):
    """Configuration for Random Forest baseline."""

    n_estimators: int = Field(default=200, ge=10, le=2000)
    max_depth: int | None = Field(default=30)
    min_samples_split: int = Field(default=5, ge=2)
    n_jobs: int = Field(default=-1)


class GNNConfig(BaseBenchConfig):
    """Configuration for GNN architectures (GCN, GraphSAGE, GAT, GIN)."""

    hidden_dim: int = Field(default=128, ge=16, le=1024)
    num_layers: int = Field(default=2, ge=1, le=10)
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)
    use_batch_norm: bool = True
    activation: str = "relu"

    # Architecture specific
    gat_heads: int = Field(default=4, ge=1, le=32)
    sage_aggr: str = Field(default="mean")
    gin_eps: float = Field(default=0.0)


class TrainingConfig(BaseBenchConfig):
    """Optimization and training loop hyperparameters."""

    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    max_epochs: int = Field(default=200, ge=1, le=2000)
    batch_size: int = Field(default=4096, ge=16, le=100000)
    patience: int = Field(default=15, ge=1, le=100)
    min_delta: float = Field(default=1e-4)
    seed: int = 42
    device: str = "cpu"


class BenchmarkRunConfig(BaseBenchConfig):
    """Complete specification for a single benchmark evaluation run."""

    model_type: ModelType = ModelType.MLP
    dataset_name: str
    split_id: str
    graph_name: str = "none"
    mlp_config: MLPConfig | None = None
    rf_config: RandomForestConfig | None = None
    gnn_config: GNNConfig | None = None
    training: TrainingConfig = TrainingConfig()
    extra_params: dict[str, Any] = Field(default_factory=dict)
