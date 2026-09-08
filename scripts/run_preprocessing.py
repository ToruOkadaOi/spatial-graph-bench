"""Run feature preprocessing pipelines Version A and Version B and issue feature manifests."""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatial_graph_bench.config.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipelineVersion,
)
from spatial_graph_bench.preprocessing.audit import verify_spatial_ignorance
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.run_preprocessing")


def process_merfish_features(split_id: str = "mouse_held_out_canonical") -> None:
    paths = ArtifactPaths.default()
    raw_path = paths.raw_data_dir / "merfish_spinal_cord" / "MERFISH_spinal_cord_resolved_0718.h5ad"
    split_file = paths.dataset_split_file("merfish_mouse_spinal_cord", split_id)

    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw MERFISH file not found: {raw_path}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    split = SplitDefinition.load_json(split_file)
    logger.info("Loading MERFISH raw file: %s", raw_path)
    adata = ad.read_h5ad(raw_path)

    # Filter to split cells
    all_split_cells = set(split.train_cell_ids + split.val_cell_ids + split.test_cell_ids)
    obs_sub = adata.obs.loc[[cid for cid in adata.obs_names if cid in all_split_cells]].copy()
    adata_sub = adata[obs_sub.index].copy()

    X_counts = adata_sub.X
    spatial_coords = adata_sub.obsm["spatial"].astype(np.float32)

    # 1. Pipeline Version A (Strict Spatial Ignorance)
    config_a = PreprocessingConfig(
        version=PreprocessingPipelineVersion.STRICT_A,
        n_pca_components=50,
        n_hvg=500,  # MERFISH panel has 500 genes total
    )

    logger.info("Executing mandatory coordinate-shuffle test for Version A...")
    audit_passed = verify_spatial_ignorance(
        X_counts=X_counts,
        obs=obs_sub,
        spatial_coords=spatial_coords,
        split=split,
        config=config_a,
        label_col="MERFISH cell type annotation",
        section_col="Section ID",
    )
    if not audit_passed:
        raise RuntimeError(
            "Coordinate-shuffle audit FAILED! Spatial information leaked into features."
        )

    logger.info("Coordinate-shuffle test passed! Building Version A feature bundle...")
    bundle_a = run_feature_pipeline(
        X_counts=X_counts,
        obs=obs_sub,
        spatial_coords=spatial_coords,
        split=split,
        config=config_a,
        label_col="MERFISH cell type annotation",
        section_col="Section ID",
    )

    out_dir_a = paths.dataset_preprocessed_dir("merfish_mouse_spinal_cord", split_id)
    bundle_a.save(out_dir_a)
    logger.info("Saved Version A preprocessed bundle: %s", out_dir_a)
    logger.info("Feature manifest hash: %s", bundle_a.manifest.compute_manifest_hash())


def process_stereoseq_features(split_id: str = "replicate_held_out_canonical") -> None:
    paths = ArtifactPaths.default()
    split_file = paths.dataset_split_file("stereoseq_axolotl_telencephalon", split_id)

    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    split = SplitDefinition.load_json(split_file)
    logger.info("Loading Stereo-seq split: %s", split_id)

    # Determine participating stages from split groups
    participating_stages = sorted(set(split.train_groups + split.val_groups + split.test_groups))
    logger.info("Participating stages: %s", participating_stages)

    adatas: dict[str, ad.AnnData] = {}
    for stage in participating_stages:
        stage_file = paths.raw_data_dir / "axolotl_telencephalon" / f"{stage}.h5ad"
        if not stage_file.is_file():
            raise FileNotFoundError(f"Raw Stereo-seq file not found for {stage}: {stage_file}")
        logger.info("Loading Stereo-seq raw file for %s: %s", stage, stage_file)
        adatas[stage] = ad.read_h5ad(stage_file)

    # Find common genes across all participating stages
    common_genes = sorted(set.intersection(*[set(a.var_names) for a in adatas.values()]))
    logger.info("Identified %d common genes across %s", len(common_genes), participating_stages)

    all_obs: list[pd.DataFrame] = []
    all_counts = []
    all_coords = []

    for stage in participating_stages:
        a_sub = adatas[stage][:, common_genes]
        counts = a_sub.layers.get("counts", a_sub.X)
        c_csr = counts if sparse.issparse(counts) else sparse.csr_matrix(counts)

        obs = a_sub.obs.copy()
        obs["slice"] = stage
        obs["global_cell_id"] = [f"{stage}_{cid}" for cid in obs.index]
        obs.index = obs["global_cell_id"]

        all_obs.append(obs)
        all_counts.append(c_csr)
        all_coords.append(a_sub.obsm["spatial"].astype(np.float32))

    joint_obs = pd.concat(all_obs)
    joint_counts = sparse.vstack(all_counts, format="csr")
    joint_coords = np.vstack(all_coords)

    # Filter to split cells
    all_split_cells = set(split.train_cell_ids + split.val_cell_ids + split.test_cell_ids)
    keep_indices = [idx for idx, cid in enumerate(joint_obs.index) if cid in all_split_cells]

    obs_sub = joint_obs.iloc[keep_indices].copy()
    counts_sub = joint_counts[keep_indices]
    spatial_sub = joint_coords[keep_indices]

    # Pipeline Version A (Strict Spatial Ignorance)
    config_a = PreprocessingConfig(
        version=PreprocessingPipelineVersion.STRICT_A,
        n_pca_components=50,
        n_hvg=min(2000, len(common_genes)),
    )

    logger.info("Executing mandatory coordinate-shuffle test for Version A...")
    audit_passed = verify_spatial_ignorance(
        X_counts=counts_sub,
        obs=obs_sub,
        spatial_coords=spatial_sub,
        split=split,
        config=config_a,
        label_col="Annotation",
        section_col="slice",
    )
    if not audit_passed:
        raise RuntimeError(
            "Coordinate-shuffle audit FAILED! Spatial information leaked into features."
        )

    logger.info("Coordinate-shuffle test passed! Building Version A feature bundle...")
    bundle_a = run_feature_pipeline(
        X_counts=counts_sub,
        obs=obs_sub,
        spatial_coords=spatial_sub,
        split=split,
        config=config_a,
        label_col="Annotation",
        section_col="slice",
    )

    out_dir_a = paths.dataset_preprocessed_dir("stereoseq_axolotl_telencephalon", split_id)
    bundle_a.save(out_dir_a)
    logger.info("Saved Version A preprocessed bundle: %s", out_dir_a)
    logger.info("Feature manifest hash: %s", bundle_a.manifest.compute_manifest_hash())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", type=str, default="mouse_held_out_canonical")
    args = parser.parse_args()

    if args.dataset == "merfish_mouse_spinal_cord":
        process_merfish_features(split_id=args.split)
    elif args.dataset == "stereoseq_axolotl_telencephalon":
        process_stereoseq_features(split_id=args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
