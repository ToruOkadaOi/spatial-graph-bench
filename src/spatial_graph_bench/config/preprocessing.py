"""Feature preprocessing pipeline configuration schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from spatial_graph_bench.config.base import BaseBenchConfig


class PreprocessingPipelineVersion(StrEnum):
    """Pipeline flavor for baseline features."""

    STRICT_A = "strict_version_a"  # Raw counts -> log-norm -> HVG -> PCA (no spatial ops)
    PLATFORM_DEFAULT_B = "platform_default_b"  # Deposited matrices as-is


class PreprocessingConfig(BaseBenchConfig):
    """Configuration for feature preprocessing."""

    version: PreprocessingPipelineVersion = PreprocessingPipelineVersion.STRICT_A
    target_sum: float = Field(
        default=1e4,
        description="Target library size for count normalization.",
    )
    n_hvg: int = Field(
        default=2000,
        description="Number of highly variable genes selected strictly on training partition.",
    )
    hvg_flavor: str = Field(
        default="seurat",
        description="HVG selection algorithm ('seurat' on log1p normalized counts).",
    )
    n_pca_components: int = Field(
        default=50,
        description="Number of principal components fit on training partition.",
    )
    scale_features: bool = Field(
        default=True,
        description="Whether to zero-center and scale features before PCA.",
    )
