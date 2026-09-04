"""Graph construction configuration schemas and protocol variants."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from spatial_graph_bench.config.base import BaseBenchConfig


class ProtocolVariant(StrEnum):
    """Evaluation protocol connectivity semantics."""

    CANONICAL_SECTION_OWN = "canonical_section_own"
    VARIANT_A_BIPARTITE = "variant_a_bipartite"
    VARIANT_B_POOLED = "variant_b_pooled"


class GraphBuilderType(StrEnum):
    """Supported graph builder algorithms."""

    SPATIAL_KNN = "spatial_knn"
    REWIRED_CONTROL = "rewired_control"
    COORDINATE_SHUFFLE = "coordinate_shuffle"
    DELAUNAY = "delaunay"
    BIPARTITE_REFERENCE = "bipartite_reference"


class EdgeWeightingMode(StrEnum):
    """Edge weighting schemes."""

    UNWEIGHTED = "unweighted"
    DISTANCE_INVERSE = "distance_inverse"
    RBF_GAUSSIAN = "rbf_gaussian"


class SpatialkNNConfig(BaseBenchConfig):
    """Configuration for physical-space coordinate k-NN graphs."""

    k: int = Field(default=6, ge=1, le=100, description="Number of spatial nearest neighbors.")
    metric: str = Field(
        default="euclidean", description="Distance metric in physical coordinate space."
    )
    symmetrize: bool = Field(default=True, description="Whether to symmetrize adjacency matrix.")
    weighting: EdgeWeightingMode = Field(
        default=EdgeWeightingMode.UNWEIGHTED,
        description="Edge weighting method.",
    )
    max_distance_cutoff: float | None = Field(
        default=None,
        description="Optional maximum physical distance cutoff preventing spurious connections across empty space.",
    )


class RewiredControlConfig(BaseBenchConfig):
    """Configuration for degree-preserving randomized rewiring negative control."""

    reference_graph_name: str = Field(
        default="spatial_knn_k6",
        description="Base spatial graph to rewire.",
    )
    n_swaps_factor: float = Field(
        default=10.0,
        description="Multiplier of total edge count for number of edge swaps.",
    )
    seed: int = Field(default=42, description="Random seed for rewiring permutation.")


class CoordinateShuffleConfig(BaseBenchConfig):
    """Configuration for coordinate-shuffled spatial graph control."""

    reference_graph_name: str = Field(
        default="spatial_knn_k6",
        description="Base spatial graph configuration to rebuild on shuffled coordinates.",
    )
    seed: int = Field(default=42, description="Random seed for coordinate permutation.")


class GraphConfig(BaseBenchConfig):
    """Composite graph construction configuration."""

    builder_type: GraphBuilderType = GraphBuilderType.SPATIAL_KNN
    protocol_variant: ProtocolVariant = ProtocolVariant.CANONICAL_SECTION_OWN
    spatial_knn: SpatialkNNConfig | None = SpatialkNNConfig()
    rewired_control: RewiredControlConfig | None = None
    coordinate_shuffle: CoordinateShuffleConfig | None = None
