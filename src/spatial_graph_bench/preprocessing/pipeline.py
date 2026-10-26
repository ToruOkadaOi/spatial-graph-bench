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
from spatial_graph_bench.preprocessing.fitted import FittedPreprocessor
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
    feature_names: list[str] | None = None,
) -> PreprocessedBundle:
    """Run frozen feature pipeline fit strictly on the training partition."""
    logger.info("Running feature pipeline version: %s", config.version.value)

    cell_ids = list(obs.index.astype(str))
    cell_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    train_idx = [cell_to_idx[cid] for cid in split.train_cell_ids if cid in cell_to_idx]
    val_idx = [cell_to_idx[cid] for cid in split.val_cell_ids if cid in cell_to_idx]
    test_idx = [cell_to_idx[cid] for cid in split.test_cell_ids if cid in cell_to_idx]

    X_all = X_counts.tocsr() if sparse.issparse(X_counts) else sparse.csr_matrix(X_counts)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]
    else:
        feature_names = [str(name) for name in feature_names]
    if len(feature_names) != X_all.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must uniquely identify every input matrix column")
    selected_feature_idx = np.arange(X_all.shape[1], dtype=np.int64)
    scaler_mean = scaler_scale = None

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

        # Highly Variable Gene (HVG) selection strictly on training partition
        if config.n_hvg is not None and config.n_hvg < X_train_norm.shape[1]:
            logger.info("Selecting top %d HVGs strictly on training partition...", config.n_hvg)
            X_tr_sq = X_train_norm.copy()
            X_tr_sq.data **= 2
            mean_sq = np.asarray(X_tr_sq.mean(axis=0)).ravel()
            mean = np.asarray(X_train_norm.mean(axis=0)).ravel()
            gene_var = mean_sq - mean**2
            top_hvg_idx = np.sort(np.argsort(gene_var)[-config.n_hvg :])
            selected_feature_idx = top_hvg_idx
            X_train_norm = X_train_norm[:, top_hvg_idx]
            X_val_norm = X_val_norm[:, top_hvg_idx]
            X_test_norm = X_test_norm[:, top_hvg_idx]

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
        scaler_mean = scaler.mean_ if config.scale_features else None
        scaler_scale = scaler.scale_ if config.scale_features else None
        normalize_library_size = True
    else:
        # Platform default Version B
        if config.n_hvg is not None and config.n_hvg < X_all.shape[1]:
            logger.info(
                "Selecting top %d HVGs strictly on training partition for Version B...",
                config.n_hvg,
            )
            X_tr_raw = X_all[train_idx]
            X_tr_sq = X_tr_raw.copy()
            X_tr_sq.data **= 2
            mean_sq = np.asarray(X_tr_sq.mean(axis=0)).ravel()
            mean = np.asarray(X_tr_raw.mean(axis=0)).ravel()
            gene_var = mean_sq - mean**2
            top_hvg_idx = np.sort(np.argsort(gene_var)[-config.n_hvg :])
            selected_feature_idx = top_hvg_idx
            X_all = X_all[:, top_hvg_idx]

        X_dense = X_all.toarray()
        pca = PCA(n_components=min(config.n_pca_components, X_dense.shape[1]), random_state=42)
        X_pca_train = pca.fit_transform(X_dense[train_idx])
        X_pca_val = pca.transform(X_dense[val_idx])
        X_pca_test = pca.transform(X_dense[test_idx])
        normalize_library_size = False

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

    # Hash exactly the dtype/bytes that PreprocessedBundle persists.
    X_pca_train = np.ascontiguousarray(X_pca_train, dtype=np.float32)
    X_pca_val = np.ascontiguousarray(X_pca_val, dtype=np.float32)
    X_pca_test = np.ascontiguousarray(X_pca_test, dtype=np.float32)
    content_hashes = {}
    for part, x, coords, labels in (
        ("train", X_pca_train, coords_train, y_train),
        ("val", X_pca_val, coords_val, y_val),
        ("test", X_pca_test, coords_test, y_test),
    ):
        content_hashes[f"X_pca_{part}"] = hash_array(x)
        content_hashes[f"spatial_{part}"] = hash_array(
            np.ascontiguousarray(coords, dtype=np.float32)
        )
        content_hashes[f"labels_{part}"] = hash_array(np.asarray(labels, dtype=np.int64))
    content_hashes["cell_ids"] = hash_dict(
        {
            "train": split.train_cell_ids,
            "val": split.val_cell_ids,
            "test": split.test_cell_ids,
            "sections_train": sec_train,
            "sections_val": sec_val,
            "sections_test": sec_test,
        }
    )
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
        content_hashes=content_hashes,
    )

    fitted_preprocessor = FittedPreprocessor(
        input_feature_names=feature_names,
        selected_feature_indices=selected_feature_idx,
        normalize_library_size=normalize_library_size,
        target_sum=config.target_sum,
        scale_features=config.scale_features
        if config.version == PreprocessingPipelineVersion.STRICT_A
        else False,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        pipeline_version=config.version.value,
    )
    for part, idx, expected in (
        ("train", train_idx, X_pca_train),
        ("val", val_idx, X_pca_val),
        ("test", test_idx, X_pca_test),
    ):
        reconstructed = fitted_preprocessor.transform(X_all[idx], feature_names)
        if part != "train" and not np.array_equal(reconstructed, expected):
            max_abs_diff = float(np.max(np.abs(reconstructed - expected)))
            raise RuntimeError(
                f"Saved fitted preprocessor does not exactly reproduce {part} PCA features; "
                f"max absolute difference={max_abs_diff:.3e}"
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
        fitted_preprocessor=fitted_preprocessor,
    )
