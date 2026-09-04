"""Evaluation summary and metrics container."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationSummary(BaseModel):
    """Summary of evaluation metrics computed on a partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition: str
    num_samples: int
    macro_f1: float
    balanced_accuracy: float
    evaluated_labels: list[str] = Field(default_factory=list)
    excluded_labels: list[str] = Field(default_factory=list)
    label_coverage: float = 1.0
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    per_section_macro_f1: dict[str, float] = Field(default_factory=dict)
    interior_macro_f1: float | None = None
    boundary_macro_f1: float | None = None
    confusion_matrix: list[list[int]] | None = None
