"""Unit tests for feature preprocessing and coordinate-shuffle audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from spatial_graph_bench.config.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipelineVersion,
)
from spatial_graph_bench.config.split import SplitConfig, SplitHierarchy
from spatial_graph_bench.preprocessing.audit import verify_spatial_ignorance
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.generator import create_split_definition


def test_preprocessing_and_spatial_ignorance_audit():
    n_cells = 30
    n_genes = 20
    rng = np.random.default_rng(42)

    X_counts = sparse.csr_matrix(rng.poisson(lam=3.0, size=(n_cells, n_genes)))
    spatial_coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "donor": ["d1"] * 15 + ["d2"] * 7 + ["d3"] * 8,
            "section_id": ["sec1"] * 15 + ["sec2"] * 7 + ["sec3"] * 8,
            "cell_type": ["TypeA", "TypeB"] * 15,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    split_cfg = SplitConfig(
        dataset_name="synth",
        split_id="s1",
        hierarchy=SplitHierarchy.DONOR_HELD_OUT,
        train_groups=["d1"],
        val_groups=["d2"],
        test_groups=["d3"],
        group_column="donor",
    )
    split = create_split_definition(obs, split_cfg)

    prep_cfg = PreprocessingConfig(
        version=PreprocessingPipelineVersion.STRICT_A,
        n_pca_components=5,
    )

    bundle = run_feature_pipeline(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=spatial_coords,
        split=split,
        config=prep_cfg,
    )

    assert bundle.X_pca_train.shape == (15, 5)
    assert bundle.X_pca_val.shape == (7, 5)
    assert bundle.X_pca_test.shape == (8, 5)
    assert bundle.spatial_train.shape == (15, 2)

    # Mandatory coordinate-shuffle test
    audit_passed = verify_spatial_ignorance(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=spatial_coords,
        split=split,
        config=prep_cfg,
    )
    assert audit_passed is True


def test_preprocessing_with_hvg_selection():
    n_cells = 30
    n_genes = 20
    rng = np.random.default_rng(42)

    X_counts = sparse.csr_matrix(rng.poisson(lam=3.0, size=(n_cells, n_genes)))
    spatial_coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "donor": ["d1"] * 15 + ["d2"] * 7 + ["d3"] * 8,
            "section_id": ["sec1"] * 15 + ["sec2"] * 7 + ["sec3"] * 8,
            "cell_type": ["TypeA", "TypeB"] * 15,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    split_cfg = SplitConfig(
        dataset_name="synth",
        split_id="s1",
        hierarchy=SplitHierarchy.DONOR_HELD_OUT,
        train_groups=["d1"],
        val_groups=["d2"],
        test_groups=["d3"],
        group_column="donor",
    )
    split = create_split_definition(obs, split_cfg)

    prep_cfg = PreprocessingConfig(
        version=PreprocessingPipelineVersion.STRICT_A,
        n_pca_components=5,
        n_hvg=10,  # 10 HVGs out of 20 genes
    )

    bundle = run_feature_pipeline(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=spatial_coords,
        split=split,
        config=prep_cfg,
    )

    assert bundle.X_pca_train.shape == (15, 5)
    assert bundle.X_pca_val.shape == (7, 5)
    assert bundle.X_pca_test.shape == (8, 5)

    # Mandatory coordinate-shuffle test with HVG selection
    audit_passed = verify_spatial_ignorance(
        X_counts=X_counts,
        obs=obs,
        spatial_coords=spatial_coords,
        split=split,
        config=prep_cfg,
    )
    assert audit_passed is True
