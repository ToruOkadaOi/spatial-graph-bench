"""Graph builder registry."""

from __future__ import annotations

from spatial_graph_bench.graph.base import BaseGraphBuilder
from spatial_graph_bench.graph.bipartite import BipartiteReferenceGraphBuilder
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder

BUILDER_REGISTRY: dict[str, type[BaseGraphBuilder]] = {
    "spatial_knn": SpatialkNNGraphBuilder,
    "bipartite_reference": BipartiteReferenceGraphBuilder,
}


def get_graph_builder(name: str) -> type[BaseGraphBuilder]:
    if name not in BUILDER_REGISTRY:
        raise KeyError(f"Builder '{name}' not found. Registered: {list(BUILDER_REGISTRY.keys())}")
    return BUILDER_REGISTRY[name]
