"""Split definition schema, disjointness enforcement, and label-coverage tracking."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spatial_graph_bench.config.split import SplitHierarchy
from spatial_graph_bench.utils.hashing import hash_dict


class SplitDefinition(BaseModel):
    """Frozen dataset split specification.

    Persisted to splits/{dataset_name}/{split_id}.json and committed to Git.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    split_id: str
    hierarchy: SplitHierarchy
    seed: int
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    train_cell_ids: list[str]
    val_cell_ids: list[str]
    test_cell_ids: list[str]
    total_cells: int
    group_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Cell counts per group across train, val, test.",
    )
    label_support: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Cell counts per label across partitions.",
    )
    # §3.4: Pre-registered unseen-class handling
    evaluated_labels: list[str] = Field(
        default_factory=list,
        description="Intersection label space (train ∩ test) evaluated for macro-F1.",
    )
    excluded_labels: list[str] = Field(
        default_factory=list,
        description="Labels completely absent from training partition, excluded from evaluation.",
    )
    coverage_ratio: float = Field(
        default=1.0,
        description="Fraction of test classes present in training partition.",
    )
    config_hash: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self.validate_disjointness()
        return self

    def validate_disjointness(self) -> None:
        """Assert that train, validation, and test groups and cell sets are strictly disjoint."""
        train_g, val_g, test_g = set(self.train_groups), set(self.val_groups), set(self.test_groups)
        if not train_g.isdisjoint(val_g):
            raise ValueError(f"Train and Val groups overlap: {train_g & val_g}")
        if not train_g.isdisjoint(test_g):
            raise ValueError(f"Train and Test groups overlap: {train_g & test_g}")
        if not val_g.isdisjoint(test_g):
            raise ValueError(f"Val and Test groups overlap: {val_g & test_g}")

        train_c, val_c, test_c = (
            set(self.train_cell_ids),
            set(self.val_cell_ids),
            set(self.test_cell_ids),
        )
        if not train_c.isdisjoint(val_c):
            raise ValueError("Train and Val cell IDs overlap!")
        if not train_c.isdisjoint(test_c):
            raise ValueError("Train and Test cell IDs overlap!")
        if not val_c.isdisjoint(test_c):
            raise ValueError("Val and Test cell IDs overlap!")

    def compute_artifact_hash(self) -> str:
        """Compute deterministic hash of the split assignments."""
        payload = {
            "dataset_name": self.dataset_name,
            "split_id": self.split_id,
            "hierarchy": self.hierarchy.value,
            "seed": self.seed,
            "train_cell_ids": self.train_cell_ids,
            "val_cell_ids": self.val_cell_ids,
            "test_cell_ids": self.test_cell_ids,
            "train_groups": self.train_groups,
            "val_groups": self.val_groups,
            "test_groups": self.test_groups,
            "evaluated_labels": self.evaluated_labels,
            "excluded_labels": self.excluded_labels,
        }
        return hash_dict(payload)

    def save_json(self, path: Path | str) -> None:
        self.validate_disjointness()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> SplitDefinition:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        instance = cls.model_validate(data)
        instance.validate_disjointness()
        return instance
