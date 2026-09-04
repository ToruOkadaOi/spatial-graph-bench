"""Base benchmark configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from spatial_graph_bench.utils.hashing import hash_config


class BaseBenchConfig(BaseModel):
    """Immutable base configuration with cryptographic hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def compute_config_hash(self) -> str:
        return hash_config(self)
