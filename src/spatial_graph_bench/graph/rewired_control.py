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

    if base_bundle.manifest.weighting != "unweighted":
        raise ValueError("Undirected rewiring requires an unweighted reference graph")
    arcs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    if any(u == v or (v, u) not in arcs for u, v in arcs):
        raise ValueError("Reference must be symmetric and have no self edges")
    sections = base_bundle.node_section_ids
    partitions = np.zeros(base_bundle.num_nodes, dtype=np.int32)
    partitions[base_bundle.val_mask.numpy()] = 1
    partitions[base_bundle.test_mask.numpy()] = 2
    groups = {}
    for u, v in sorted(arcs):
        if partitions[u] != partitions[v] or sections[u] != sections[v]:
            raise ValueError("Reference crosses partition/section boundaries")
        if u < v:
            groups.setdefault((int(partitions[u]), sections[u]), []).append((u, v))
    rewired = set()
    successful_swaps = 0
    for edges in groups.values():
        edge_set = set(edges)
        m = len(edges)
        if m >= 2:
            for _ in range(int(m * config.n_swaps_factor)):
                i, j = rng.sample(range(m), 2)
                u, v = edges[i]
                x, y = edges[j]
                if rng.random() < 0.5:
                    x, y = y, x
                if len({u, v, x, y}) != 4:
                    continue
                a, b = tuple(sorted((u, y))), tuple(sorted((x, v)))
                if a in edge_set or b in edge_set:
                    continue
                edge_set.remove(edges[i])
                edge_set.remove(edges[j])
                edge_set.update((a, b))
                edges[i], edges[j] = a, b
                successful_swaps += 1
        for u, v in edge_set:
            rewired.update(((u, v), (v, u)))
    logger.info("Undirected rewiring: %d successful pair swaps", successful_swaps)
    new_edge_index = torch.tensor(sorted(rewired), dtype=torch.long).reshape(-1, 2).t().contiguous()

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
