"""Split configuration schemas and holdout hierarchy modes."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from spatial_graph_bench.config.base import BaseBenchConfig


class SplitHierarchy(StrEnum):
    """Hierarchy level for data partitioning."""

    DONOR_HELD_OUT = "donor-held-out"
    SECTION_HELD_OUT = "section-held-out"
    REPLICATE_HELD_OUT = "replicate-held-out"
    SPATIAL_BLOCK = "spatial-block"


class SplitConfig(BaseBenchConfig):
    """Configuration defining a dataset partitioning scheme."""

    dataset_name: str
    split_id: str
    hierarchy: SplitHierarchy
    seed: int = 42
    train_groups: list[str] = Field(default_factory=list)
    val_groups: list[str] = Field(default_factory=list)
    test_groups: list[str] = Field(default_factory=list)
    group_column: str = Field(
        default="donor_id",
        description="Metadata column used for group partitioning (e.g., Mouse ID, Section ID).",
    )
    val_ratio_within_train: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Internal validation ratio carved strictly from training groups if val_groups empty.",
    )
