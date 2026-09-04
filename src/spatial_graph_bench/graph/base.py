"""Abstract base class for graph builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle


class BaseGraphBuilder(ABC):
    """Abstract interface for benchmark graph construction."""

    @abstractmethod
    def build(
        self,
        bundle: PreprocessedBundle,
        graph_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> GraphBundle:
        """Construct graph bundle enforcing strict topological invariants."""
        raise NotImplementedError
