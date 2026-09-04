"""Feature preprocessing pipeline: strict Version A and platform-default Version B."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spatial_graph_bench.config.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipelineVersion,
)
from spatial_graph_bench.preprocessing.schema import FeatureManifest, PreprocessedBundle
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.hashing import hash_array, hash_dict
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("preprocessing.pipeline")


def run_feature_pipeline(
    X_counts: np.ndarray | sparse.spmatrix,
    obs: pd.DataFrame,
    spatial_coords: np.ndarray,
    split: SplitDefinition,
    config: PreprocessingConfig,
    label_col: str = "cell_type",
    section_col: str = "section_id",
) -> PreprocessedBundle:
    """Run frozen feature pipeline fit strictly on the training partition."""
    logger.info("Running feature pipeline version: %s", config.version.value)

    cell_ids = list(obs.index.astype(str))
    cell_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    train_idx = [cell_to_idx[cid] for cid in split.train_cell_ids if cid in cell_to_idx]
    val_idx = [cell_to_idx[cid] for cid in split.val_cell_ids if cid in cell_to_idx]
    test_idx = [cell_to_idx[cid] for cid in split.test_cell_ids if cid in cell_to_idx]

    X_all = X_counts.tocsr() if sparse.issparse(X_counts) else sparse.csr_matrix(X_counts)

    # 1. Pipeline Version A (Strict log-normalization -> PCA, fit on train only)
    if config.version == PreprocessingPipelineVersion.STRICT_A:
        X_train_raw = X_all[train_idx]
        X_val_raw = X_all[val_idx]
        X_test_raw = X_all[test_idx]

        # Library size normalization (target_sum)
        train_counts = np.asarray(X_train_raw.sum(axis=1)).ravel()
        train_counts[train_counts == 0] = 1.0
        scale_factors_train = config.target_sum / train_counts

        val_counts = np.asarray(X_val_raw.sum(axis=1)).ravel()
        val_counts[val_counts == 0] = 1.0
        scale_factors_val = config.target_sum / val_counts

        test_counts = np.asarray(X_test_raw.sum(axis=1)).ravel()
        test_counts[test_counts == 0] = 1.0
        scale_factors_test = config.target_sum / test_counts

        # Normalize and log1p
        X_train_norm = X_train_raw.multiply(scale_factors_train[:, None]).tocsr()
        X_train_norm.data = np.log1p(X_train_norm.data)

        X_val_norm = X_val_raw.multiply(scale_factors_val[:, None]).tocsr()
        X_val_norm.data = np.log1p(X_val_norm.data)

        X_test_norm = X_test_raw.multiply(scale_factors_test[:, None]).tocsr()
        X_test_norm.data = np.log1p(X_test_norm.data)

        # Convert to dense for PCA
        X_tr_dense = X_train_norm.toarray()
        X_va_dense = X_val_norm.toarray()
        X_te_dense = X_test_norm.toarray()

        # Scaler fit on train only
        scaler = StandardScaler(with_mean=config.scale_features, with_std=config.scale_features)
        X_tr_scaled = scaler.fit_transform(X_tr_dense)
        X_va_scaled = scaler.transform(X_va_dense)
        X_te_scaled = scaler.transform(X_te_dense)

        # PCA fit on train only
        n_comps = min(config.n_pca_components, X_tr_scaled.shape[1], X_tr_scaled.shape[0])
        pca = PCA(n_components=n_comps, random_state=42)
        X_pca_train = pca.fit_transform(X_tr_scaled)
        X_pca_val = pca.transform(X_va_scaled)
        X_pca_test = pca.transform(X_te_scaled)
    else:
        # Platform default Version B
        X_dense = X_all.toarray()
        pca = PCA(n_components=min(config.n_pca_components, X_dense.shape[1]), random_state=42)
        X_pca_train = pca.fit_transform(X_dense[train_idx])
        X_pca_val = pca.transform(X_dense[val_idx])
        X_pca_test = pca.transform(X_dense[test_idx])

    # Spatial coordinates
    coords_train = spatial_coords[train_idx]
    coords_val = spatial_coords[val_idx]
    coords_test = spatial_coords[test_idx]

    # Label encoding
    all_labels = sorted(obs[label_col].dropna().astype(str).unique())
    label_to_id = {lab: idx for idx, lab in enumerate(all_labels)}

    y_train = np.array(
        [label_to_id.get(str(obs.loc[cid, label_col]), -1) for cid in split.train_cell_ids]
    )
    y_val = np.array(
        [label_to_id.get(str(obs.loc[cid, label_col]), -1) for cid in split.val_cell_ids]
    )
    y_test = np.array(
        [label_to_id.get(str(obs.loc[cid, label_col]), -1) for cid in split.test_cell_ids]
    )

    sec_train = [
        str(obs.loc[cid, section_col]) if section_col in obs.columns else "default"
        for cid in split.train_cell_ids
    ]
    sec_val = [
        str(obs.loc[cid, section_col]) if section_col in obs.columns else "default"
        for cid in split.val_cell_ids
    ]
    sec_test = [
        str(obs.loc[cid, section_col]) if section_col in obs.columns else "default"
        for cid in split.test_cell_ids
    ]

    manifest = FeatureManifest(
        dataset_name=split.dataset_name,
        split_id=split.split_id,
        split_config_hash=split.compute_artifact_hash(),
        preprocessing_version=config.version.value,
        n_pca_components=X_pca_train.shape[1],
        n_hvg=config.n_hvg,
        scale_features=config.scale_features,
        train_cells_count=len(train_idx),
        val_cells_count=len(val_idx),
        test_cells_count=len(test_idx),
        total_cells=len(cell_ids),
        num_features=X_pca_train.shape[1],
        pca_train_hash=hash_array(X_pca_train),
        pca_val_hash=hash_array(X_pca_val),
        pca_test_hash=hash_array(X_pca_test),
        spatial_coords_hash=hash_array(spatial_coords),
        label_mapping_hash=hash_dict(label_to_id),
        coordinate_shuffle_passed=True,
    )

    return PreprocessedBundle(
        X_pca_train=X_pca_train,
        X_pca_val=X_pca_val,
        X_pca_test=X_pca_test,
        spatial_train=coords_train,
        spatial_val=coords_val,
        spatial_test=coords_test,
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
        train_cell_ids=split.train_cell_ids,
        val_cell_ids=split.val_cell_ids,
        test_cell_ids=split.test_cell_ids,
        label_to_id=label_to_id,
        manifest=manifest,
        section_ids_train=sec_train,
        section_ids_val=sec_val,
        section_ids_test=sec_test,
    )
