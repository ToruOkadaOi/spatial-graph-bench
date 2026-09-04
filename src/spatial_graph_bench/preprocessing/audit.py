"""Coordinate-shuffle feature audit guaranteeing strict spatial ignorance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from spatial_graph_bench.config.preprocessing import PreprocessingConfig
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("preprocessing.audit")


def verify_spatial_ignorance(
    X_counts: np.ndarray | sparse.spmatrix,
    obs: pd.DataFrame,
    spatial_coords: np.ndarray,
    split: SplitDefinition,
    config: PreprocessingConfig,
    seed: int = 42,
) -> bool:
    """Execute coordinate-shuffle test on feature preprocessing pipeline.

    Shuffles spatial coordinates and asserts that the computed feature matrix
    (X_pca) is completely invariant. If any feature changes, the baseline was
    not spatially ignorant!
    """
    logger.info("Executing coordinate-shuffle test for spatial ignorance...")

    # 1. Run pipeline on intact coordinates
    bundle_original = run_feature_pipeline(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=spatial_coords,
        split=split,
        config=config,
    )

    # 2. Permute coordinates randomly
    rng = np.random.default_rng(seed)
    shuffled_coords = rng.permutation(spatial_coords)

    # 3. Run pipeline on permuted coordinates
    bundle_shuffled = run_feature_pipeline(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=shuffled_coords,
        split=split,
        config=config,
    )

    # 4. Assert feature identity
    tr_diff = np.max(np.abs(bundle_original.X_pca_train - bundle_shuffled.X_pca_train))
    va_diff = np.max(np.abs(bundle_original.X_pca_val - bundle_shuffled.X_pca_val))
    te_diff = np.max(np.abs(bundle_original.X_pca_test - bundle_shuffled.X_pca_test))

    max_diff = max(tr_diff, va_diff, te_diff)
    if max_diff > 1e-6:
        logger.error(
            "SPATIAL LEAKAGE DETECTED! Features changed under coordinate shuffling: max diff = %e",
            max_diff,
        )
        return False

    logger.info("Spatial ignorance audit PASSED (max diff = %e <= 1e-6)", max_diff)
    return True
