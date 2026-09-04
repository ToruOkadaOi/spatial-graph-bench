"""Coordinate-shuffled negative control graph builder."""

from __future__ import annotations

import numpy as np

from spatial_graph_bench.config.graph import CoordinateShuffleConfig, SpatialkNNConfig
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("graph.coordinate_shuffle")


def build_coordinate_shuffled_graph(
    bundle: PreprocessedBundle,
    config: CoordinateShuffleConfig,
    graph_name: str,
    k: int = 6,
) -> GraphBundle:
    """Construct spatial k-NN graph on permuted physical coordinates within each partition."""
    logger.info("Building coordinate-shuffled graph (seed=%d)...", config.seed)

    rng = np.random.default_rng(config.seed)

    shuffled_sp_train = rng.permutation(bundle.spatial_train)
    shuffled_sp_val = rng.permutation(bundle.spatial_val)
    shuffled_sp_test = rng.permutation(bundle.spatial_test)

    shuffled_bundle = PreprocessedBundle(
        X_pca_train=bundle.X_pca_train,
        X_pca_val=bundle.X_pca_val,
        X_pca_test=bundle.X_pca_test,
        spatial_train=shuffled_sp_train,
        spatial_val=shuffled_sp_val,
        spatial_test=shuffled_sp_test,
        train_labels=bundle.train_labels,
        val_labels=bundle.val_labels,
        test_labels=bundle.test_labels,
        train_cell_ids=bundle.train_cell_ids,
        val_cell_ids=bundle.val_cell_ids,
        test_cell_ids=bundle.test_cell_ids,
        label_to_id=bundle.label_to_id,
        manifest=bundle.manifest,
        section_ids_train=bundle.section_ids_train,
        section_ids_val=bundle.section_ids_val,
        section_ids_test=bundle.section_ids_test,
    )

    builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=k))
    res = builder.build(shuffled_bundle, graph_name=graph_name)

    # Re-wrap manifest with coordinate_shuffle builder_type
    manifest = res.manifest.model_copy(update={"builder_type": "coordinate_shuffle"})
    return GraphBundle(
        edge_index=res.edge_index,
        num_nodes=res.num_nodes,
        train_mask=res.train_mask,
        val_mask=res.val_mask,
        test_mask=res.test_mask,
        node_cell_ids=res.node_cell_ids,
        manifest=manifest,
        edge_weight=res.edge_weight,
        node_section_ids=res.node_section_ids,
    )
