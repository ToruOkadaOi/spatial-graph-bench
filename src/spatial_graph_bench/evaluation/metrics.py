"""Metrics computation enforcing §3.4 label-space intersection rules."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from spatial_graph_bench.evaluation.schema import EvaluationSummary

# Suppress benign sklearn warning when predicted class isn't in ground truth subset
warnings.filterwarnings(
    "ignore",
    message=".*y_pred contains classes not in y_true.*",
    category=UserWarning,
)


def compute_partition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    partition: str,
    label_to_id: dict[str, int],
    evaluated_labels: list[str] | None = None,
    excluded_labels: list[str] | None = None,
    section_ids: list[str] | None = None,
    is_boundary: np.ndarray | None = None,
) -> EvaluationSummary:
    """Compute evaluation metrics enforcing §3.4 label-space intersection."""
    id_to_label = {v: k for k, v in label_to_id.items()}
    all_label_names = [id_to_label[i] for i in range(len(id_to_label))]

    if evaluated_labels is not None:
        eval_ids = [label_to_id[lab] for lab in evaluated_labels if lab in label_to_id]
        excl_labels = list(excluded_labels or [])
    else:
        # If not specified, default to unique labels in y_true
        eval_ids = sorted(set(y_true[y_true >= 0]))
        excl_labels = []

    eval_label_names = [id_to_label[i] for i in eval_ids]
    coverage = len(eval_label_names) / max(len(all_label_names), 1)

    # Invariant §3.4: Filter out cells belonging to excluded classes
    mask = np.isin(y_true, eval_ids)
    if mask.sum() == 0:
        return EvaluationSummary(
            partition=partition,
            num_samples=0,
            macro_f1=0.0,
            balanced_accuracy=0.0,
            evaluated_labels=eval_label_names,
            excluded_labels=excl_labels,
            label_coverage=coverage,
        )

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Compute overall macro-F1 over the evaluated intersection label space
    macro_f1 = float(
        f1_score(
            y_true_filtered,
            y_pred_filtered,
            labels=eval_ids,
            average="macro",
            zero_division=0.0,
        )
    )

    bal_acc = float(
        balanced_accuracy_score(
            y_true_filtered,
            y_pred_filtered,
        )
    )

    # Per-class F1
    per_class_raw = f1_score(
        y_true_filtered,
        y_pred_filtered,
        labels=eval_ids,
        average=None,
        zero_division=0.0,
    )
    per_class_f1 = {eval_label_names[i]: float(per_class_raw[i]) for i in range(len(eval_ids))}

    # Per-section macro-F1
    per_section_f1: dict[str, float] = {}
    if section_ids is not None:
        sec_arr = np.array(section_ids)[mask]
        for sec in np.unique(sec_arr):
            s_mask = sec_arr == sec
            if s_mask.sum() > 0:
                s_f1 = float(
                    f1_score(
                        y_true_filtered[s_mask],
                        y_pred_filtered[s_mask],
                        labels=eval_ids,
                        average="macro",
                        zero_division=0.0,
                    )
                )
                per_section_f1[str(sec)] = s_f1

    # Interior vs boundary stratification
    interior_f1 = None
    boundary_f1 = None
    if is_boundary is not None:
        b_arr = is_boundary[mask]
        if (~b_arr).sum() > 0:
            interior_f1 = float(
                f1_score(
                    y_true_filtered[~b_arr],
                    y_pred_filtered[~b_arr],
                    labels=eval_ids,
                    average="macro",
                    zero_division=0.0,
                )
            )
        if b_arr.sum() > 0:
            boundary_f1 = float(
                f1_score(
                    y_true_filtered[b_arr],
                    y_pred_filtered[b_arr],
                    labels=eval_ids,
                    average="macro",
                    zero_division=0.0,
                )
            )

    # Confusion matrix
    cm = confusion_matrix(
        y_true_filtered,
        y_pred_filtered,
        labels=eval_ids,
    ).tolist()

    return EvaluationSummary(
        partition=partition,
        num_samples=int(mask.sum()),
        macro_f1=macro_f1,
        balanced_accuracy=bal_acc,
        evaluated_labels=eval_label_names,
        excluded_labels=excl_labels,
        label_coverage=coverage,
        per_class_f1=per_class_f1,
        per_section_macro_f1=per_section_f1,
        interior_macro_f1=interior_f1,
        boundary_macro_f1=boundary_f1,
        confusion_matrix=cm,
    )
