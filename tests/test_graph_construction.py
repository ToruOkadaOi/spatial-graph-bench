"""Unit tests for graph construction invariants and negative controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from spatial_graph_bench.config.graph import (
    CoordinateShuffleConfig,
    RewiredControlConfig,
    SpatialkNNConfig,
)
from spatial_graph_bench.config.preprocessing import PreprocessingConfig
from spatial_graph_bench.config.split import SplitConfig, SplitHierarchy
from spatial_graph_bench.graph.bipartite import BipartiteReferenceGraphBuilder
from spatial_graph_bench.graph.coordinate_shuffle import build_coordinate_shuffled_graph
from spatial_graph_bench.graph.rewired_control import build_rewired_control_graph
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.generator import create_split_definition


def _make_dummy_bundle():
    n_cells = 40
    rng = np.random.default_rng(42)
    X = sparse.csr_matrix(rng.poisson(3.0, (n_cells, 10)))
    coords = rng.uniform(0, 100, (n_cells, 2)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "donor": ["d1"] * 20 + ["d2"] * 10 + ["d3"] * 10,
            "section": ["sec1"] * 20 + ["sec2"] * 10 + ["sec3"] * 10,
            "cell_type": ["T", "B"] * 20,
        },
        index=[f"c_{i}" for i in range(n_cells)],
    )

    split = create_split_definition(
        obs,
        SplitConfig(
            dataset_name="synth",
            split_id="s1",
            hierarchy=SplitHierarchy.DONOR_HELD_OUT,
            train_groups=["d1"],
            val_groups=["d2"],
            test_groups=["d3"],
            group_column="donor",
        ),
    )

    return run_feature_pipeline(
        X, obs, coords, split, PreprocessingConfig(n_pca_components=4), section_col="section"
    )


def test_spatial_knn_topological_invariants():
    bundle = _make_dummy_bundle()
    builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4))
    graph = builder.build(bundle, graph_name="spatial_knn_k4")

    # Assert invariant: zero cross-partition edges and zero cross-section edges
    assert graph.manifest.num_disallowed_cross_partition_edges == 0
    assert graph.manifest.num_disallowed_cross_section_edges == 0
    assert graph.edge_index.size(0) == 2

    # Verify edge index within partition bounds
    src = graph.edge_index[0].numpy()
    tgt = graph.edge_index[1].numpy()

    # Train: [0, 20), Val: [20, 30), Test: [30, 40)
    for s, t in zip(src, tgt, strict=True):
        if s < 20:
            assert t < 20  # Train to Train only
        elif 20 <= s < 30:
            assert 20 <= t < 30  # Val to Val only
        else:
            assert s >= 30 and t >= 30  # Test to Test only


def test_rewired_control_graph():
    bundle = _make_dummy_bundle()
    builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4))
    base_graph = builder.build(bundle, graph_name="base")

    rewired = build_rewired_control_graph(
        base_graph,
        RewiredControlConfig(seed=42),
        graph_name="rewired",
    )
    assert rewired.num_nodes == base_graph.num_nodes
    assert rewired.edge_index.size(1) == base_graph.edge_index.size(1)
    assert rewired.manifest.num_disallowed_cross_partition_edges == 0


def test_coordinate_shuffled_graph():
    bundle = _make_dummy_bundle()
    shuffled = build_coordinate_shuffled_graph(
        bundle,
        CoordinateShuffleConfig(seed=42),
        graph_name="shuffled",
        k=4,
    )
    assert (
        shuffled.num_nodes
        == bundle.X_pca_train.shape[0] + bundle.X_pca_val.shape[0] + bundle.X_pca_test.shape[0]
    )
    assert shuffled.manifest.num_disallowed_cross_partition_edges == 0


def test_bipartite_variant_a():
    bundle = _make_dummy_bundle()
    builder = BipartiteReferenceGraphBuilder(k=5)
    bipartite_graph = builder.build(bundle, graph_name="bipartite_k5")

    # Variant A: test/val have 0 query-query edges
    assert bipartite_graph.manifest.num_val_val_edges == 0
    assert bipartite_graph.manifest.num_test_test_edges == 0
    assert bipartite_graph.manifest.num_train_train_edges > 0
