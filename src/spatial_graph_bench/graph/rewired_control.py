"""Degree-preserving randomized rewiring negative control."""

from __future__ import annotations

import random

import numpy as np
import torch

from spatial_graph_bench.config.graph import RewiredControlConfig
from spatial_graph_bench.graph.schema import GraphBundle, GraphManifest
from spatial_graph_bench.utils.hashing import hash_array
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("graph.rewired_control")


def build_rewired_control_graph(
    base_bundle: GraphBundle,
    config: RewiredControlConfig,
    graph_name: str,
) -> GraphBundle:
    """Construct degree-preserving rewired graph within each partition/section."""
    rng = random.Random(config.seed)
    edge_index = base_bundle.edge_index.clone().cpu().numpy()

    if edge_index.shape[1] == 0:
        return base_bundle

    # Map edges to list of (u, v)
    edges = list(zip(edge_index[0], edge_index[1], strict=True))
    edge_set = set(edges)
    m = len(edges)
    n_swaps = int(m * config.n_swaps_factor)

    logger.info("Rewiring %d edges with %d attempted swaps (seed=%d)...", m, n_swaps, config.seed)

    # Invariant: rewiring must preserve partition and section disjointness
    sections = base_bundle.node_section_ids
    partitions = np.zeros(base_bundle.num_nodes, dtype=np.int32)
    partitions[base_bundle.val_mask.numpy()] = 1
    partitions[base_bundle.test_mask.numpy()] = 2

    successful_swaps = 0
    for _ in range(n_swaps):
        i = rng.randrange(m)
        j = rng.randrange(m)
        if i == j:
            continue

        u, v = edges[i]
        x, y = edges[j]

        # Invariant: can only swap if both edges belong to the same partition and section!
        if partitions[u] != partitions[x] or sections[u] != sections[x]:
            continue

        # Propose (u, y) and (x, v)
        if u == y or x == v:
            continue
        if (u, y) in edge_set or (x, v) in edge_set:
            continue

        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))
        edges[i] = (u, y)
        edges[j] = (x, v)
        successful_swaps += 1

    logger.info("Rewiring completed: %d successful edge swaps executed.", successful_swaps)

    sorted_edges = sorted(edges)
    new_edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()

    manifest = GraphManifest(
        graph_name=graph_name,
        builder_type="rewired_control",
        dataset_name=base_bundle.manifest.dataset_name,
        split_id=base_bundle.manifest.split_id,
        protocol_variant=base_bundle.manifest.protocol_variant,
        k=base_bundle.manifest.k,
        metric=base_bundle.manifest.metric,
        weighting=base_bundle.manifest.weighting,
        edge_index_convention=base_bundle.manifest.edge_index_convention,
        num_nodes=base_bundle.num_nodes,
        num_edges=new_edge_index.size(1),
        num_train_nodes=base_bundle.manifest.num_train_nodes,
        num_val_nodes=base_bundle.manifest.num_val_nodes,
        num_test_nodes=base_bundle.manifest.num_test_nodes,
        num_train_train_edges=base_bundle.manifest.num_train_train_edges,
        num_val_val_edges=base_bundle.manifest.num_val_val_edges,
        num_test_test_edges=base_bundle.manifest.num_test_test_edges,
        num_disallowed_cross_partition_edges=0,
        num_disallowed_cross_section_edges=0,
        edge_index_hash=hash_array(new_edge_index.numpy()),
        edge_weight_hash=None,
        feature_manifest_hash=base_bundle.manifest.feature_manifest_hash,
    )

    return GraphBundle(
        edge_index=new_edge_index,
        num_nodes=base_bundle.num_nodes,
        train_mask=base_bundle.train_mask,
        val_mask=base_bundle.val_mask,
        test_mask=base_bundle.test_mask,
        node_cell_ids=base_bundle.node_cell_ids,
        manifest=manifest,
        edge_weight=None,
        node_section_ids=base_bundle.node_section_ids,
    )
