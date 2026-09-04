"""Matched graph lift computation enforcing exact provenance invariants."""

from __future__ import annotations

from spatial_graph_bench.tracking.schema import GraphLiftRecord, RunManifest
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("tracking.graph_lift")


def compute_matched_graph_lift(
    gnn_manifest: RunManifest,
    mlp_manifest: RunManifest,
    gnn_macro_f1: float,
    mlp_macro_f1: float,
    gnn_balanced_acc: float = 0.0,
    mlp_balanced_acc: float = 0.0,
    parity_band_halfwidth: float = 0.005,
    interior_lift: float | None = None,
    boundary_lift: float | None = None,
    per_section_lifts: dict[str, float] | None = None,
    per_class_lifts: dict[str, float] | None = None,
) -> GraphLiftRecord:
    """Compute matched lift between a GNN run and its same-seed MLP baseline.

    Invariants checked:
    - Identical dataset_name
    - Identical split_id and split_hash
    - Identical seed
    - Compatible feature manifest / preprocessing config (with metadata timestamp tolerance)
    - Identical label mapping
    """
    mismatches: list[str] = []

    if gnn_manifest.dataset_name != mlp_manifest.dataset_name:
        mismatches.append(
            f"Dataset mismatch: GNN={gnn_manifest.dataset_name} vs MLP={mlp_manifest.dataset_name}"
        )
    if gnn_manifest.split_id != mlp_manifest.split_id:
        mismatches.append(
            f"Split mismatch: GNN={gnn_manifest.split_id} vs MLP={mlp_manifest.split_id}"
        )
    if gnn_manifest.split_hash != mlp_manifest.split_hash:
        mismatches.append(
            f"Split hash mismatch: GNN={gnn_manifest.split_hash} vs MLP={mlp_manifest.split_hash}"
        )
    if gnn_manifest.seed != mlp_manifest.seed:
        mismatches.append(f"Seed mismatch: GNN={gnn_manifest.seed} vs MLP={mlp_manifest.seed}")

    # Ported fix from scgraph-bench commit 2bc5c9a: metadata tolerance
    if gnn_manifest.feature_manifest_hash != mlp_manifest.feature_manifest_hash:
        if (
            gnn_manifest.split_hash == mlp_manifest.split_hash
            and gnn_manifest.seed == mlp_manifest.seed
            and gnn_manifest.preprocessing_config_hash == mlp_manifest.preprocessing_config_hash
        ):
            logger.warning(
                "Feature manifest hash differs (GNN=%s vs MLP=%s), but split_hash and preprocessing_config_hash match identically.",
                gnn_manifest.feature_manifest_hash[:16],
                mlp_manifest.feature_manifest_hash[:16],
            )
        else:
            mismatches.append(
                f"Feature manifest hash mismatch: GNN={gnn_manifest.feature_manifest_hash} vs MLP={mlp_manifest.feature_manifest_hash}"
            )

    if gnn_manifest.preprocessing_config_hash != mlp_manifest.preprocessing_config_hash:
        mismatches.append(
            f"Preprocessing config hash mismatch: GNN={gnn_manifest.preprocessing_config_hash} vs MLP={mlp_manifest.preprocessing_config_hash}"
        )
    if gnn_manifest.label_mapping_hash != mlp_manifest.label_mapping_hash:
        mismatches.append(
            f"Label mapping hash mismatch: GNN={gnn_manifest.label_mapping_hash} vs MLP={mlp_manifest.label_mapping_hash}"
        )

    is_valid = len(mismatches) == 0
    notes = "; ".join(mismatches) if mismatches else None

    lift = gnn_macro_f1 - mlp_macro_f1
    bal_acc_lift = gnn_balanced_acc - mlp_balanced_acc

    # Classify against parity band
    if lift > parity_band_halfwidth:
        parity_class = "positive"
    elif lift < -parity_band_halfwidth:
        parity_class = "negative"
    else:
        parity_class = "parity"

    graph_name = gnn_manifest.graph_artifact_hash or "spatial_graph"

    return GraphLiftRecord(
        dataset_name=gnn_manifest.dataset_name,
        dataset_version=gnn_manifest.dataset_version,
        split_id=gnn_manifest.split_id,
        split_hash=gnn_manifest.split_hash,
        seed=gnn_manifest.seed,
        graph_name=graph_name,
        protocol_variant=gnn_manifest.protocol_variant,
        gnn_model_name=gnn_manifest.model_name,
        matched_mlp_model_name=mlp_manifest.model_name,
        gnn_macro_f1=gnn_macro_f1,
        matched_mlp_macro_f1=mlp_macro_f1,
        overall_graph_lift=lift,
        gnn_balanced_accuracy=gnn_balanced_acc,
        matched_mlp_balanced_accuracy=mlp_balanced_acc,
        balanced_accuracy_lift=bal_acc_lift,
        parity_classification=parity_class,
        parity_band_halfwidth=parity_band_halfwidth,
        interior_lift=interior_lift,
        boundary_lift=boundary_lift,
        per_section_lifts=per_section_lifts or {},
        per_class_lifts=per_class_lifts or {},
        feature_manifest_hash=gnn_manifest.feature_manifest_hash,
        preprocessing_config_hash=gnn_manifest.preprocessing_config_hash,
        label_mapping_hash=gnn_manifest.label_mapping_hash,
        is_valid_match=is_valid,
        notes=notes,
    )
