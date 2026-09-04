"""Physical coordinate k-NN graph builder enforcing section-own inductive topology."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from spatial_graph_bench.config.graph import EdgeWeightingMode, SpatialkNNConfig
from spatial_graph_bench.graph.base import BaseGraphBuilder
from spatial_graph_bench.graph.schema import GraphBundle, GraphManifest
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.hashing import hash_array
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("graph.spatial_knn")


class SpatialkNNGraphBuilder(BaseGraphBuilder):
    """Constructs physical-space k-NN graphs strictly partitioned by section and split."""

    def __init__(self, config: SpatialkNNConfig | None = None) -> None:
        self.config = config or SpatialkNNConfig()

    def build(
        self,
        bundle: PreprocessedBundle,
        graph_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> GraphBundle:
        _ = extra_metadata
        k = self.config.k
        metric = self.config.metric
        weighting = self.config.weighting

        n_tr = len(bundle.train_cell_ids)
        n_va = len(bundle.val_cell_ids)
        n_te = len(bundle.test_cell_ids)
        n_total = n_tr + n_va + n_te

        logger.info(
            "Building section-own spatial k-NN graph (k=%d, metric='%s', weighting='%s') for %d nodes...",
            k,
            metric,
            weighting.value,
            n_total,
        )

        # Global node ordering: train, val, test
        cell_ids = bundle.train_cell_ids + bundle.val_cell_ids + bundle.test_cell_ids
        sections = bundle.section_ids_train + bundle.section_ids_val + bundle.section_ids_test
        coords = np.vstack([bundle.spatial_train, bundle.spatial_val, bundle.spatial_test])

        train_mask = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_tr] = True

        val_mask = torch.zeros(n_total, dtype=torch.bool)
        val_mask[n_tr : n_tr + n_va] = True

        test_mask = torch.zeros(n_total, dtype=torch.bool)
        test_mask[n_tr + n_va :] = True

        # Partitions: 0=train, 1=val, 2=test
        partitions = np.zeros(n_total, dtype=np.int32)
        partitions[n_tr : n_tr + n_va] = 1
        partitions[n_tr + n_va :] = 2

        edges: set[tuple[int, int]] = set()
        edge_weights_dict: dict[tuple[int, int], float] = {}

        # Canonical Section-Own Subgraph Inductive Rule:
        # Group cells by (partition, section_id)
        # Graph construction is executed strictly WITHIN each (partition, section) subgraph!
        partition_section_pairs = sorted(set(zip(partitions, sections, strict=True)))

        num_tr_tr = 0
        num_va_va = 0
        num_te_te = 0

        for part_id, sec_id in partition_section_pairs:
            part_sec_indices = [
                idx
                for idx in range(n_total)
                if partitions[idx] == part_id and sections[idx] == sec_id
            ]

            n_sec_nodes = len(part_sec_indices)
            if n_sec_nodes <= 1:
                continue

            sub_coords = coords[part_sec_indices]
            eff_k = min(k, n_sec_nodes - 1)

            nn = NearestNeighbors(n_neighbors=eff_k + 1, metric=metric)
            nn.fit(sub_coords)
            dists, indices = nn.kneighbors(sub_coords)

            for local_i in range(n_sec_nodes):
                global_i = part_sec_indices[local_i]
                for rank in range(1, eff_k + 1):
                    local_j = int(indices[local_i, rank])
                    global_j = part_sec_indices[local_j]
                    d = float(dists[local_i, rank])

                    if self.config.max_distance_cutoff and d > self.config.max_distance_cutoff:
                        continue

                    edge = (global_j, global_i)  # source -> target
                    edges.add(edge)
                    edge_weights_dict[edge] = d

                    if self.config.symmetrize:
                        rev_edge = (global_i, global_j)
                        edges.add(rev_edge)
                        edge_weights_dict[rev_edge] = d

        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight_tensor = None
            num_tr_tr = 0
            num_va_va = 0
            num_te_te = 0
        else:
            edge_list = sorted(edges)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            num_tr_tr = sum(1 for src, _ in edges if partitions[src] == 0)
            num_va_va = sum(1 for src, _ in edges if partitions[src] == 1)
            num_te_te = sum(1 for src, _ in edges if partitions[src] == 2)

            if weighting == EdgeWeightingMode.UNWEIGHTED:
                edge_weight_tensor = torch.ones(len(edge_list), dtype=torch.float32)
            elif weighting == EdgeWeightingMode.DISTANCE_INVERSE:
                raw_dists = np.array([edge_weights_dict[e] for e in edge_list], dtype=np.float32)
                edge_weight_tensor = torch.from_numpy(1.0 / (raw_dists + 1e-5))
            else:
                raw_dists = np.array([edge_weights_dict[e] for e in edge_list], dtype=np.float32)
                sigma = float(np.median(raw_dists)) if len(raw_dists) > 0 else 1.0
                sigma = max(sigma, 1e-5)
                rbf = np.exp(-0.5 * (raw_dists / sigma) ** 2)
                edge_weight_tensor = torch.from_numpy(rbf.astype(np.float32))

        # Assert zero disallowed edges
        num_disallowed_part = 0
        num_disallowed_sec = 0
        for src, tgt in edges:
            if partitions[src] != partitions[tgt]:
                num_disallowed_part += 1
            if sections[src] != sections[tgt]:
                num_disallowed_sec += 1

        if num_disallowed_part > 0 or num_disallowed_sec > 0:
            raise RuntimeError(
                f"Topological invariant violated! Disallowed cross-partition: {num_disallowed_part}, "
                f"cross-section: {num_disallowed_sec}"
            )

        manifest = GraphManifest(
            graph_name=graph_name,
            builder_type="spatial_knn",
            dataset_name=bundle.manifest.dataset_name,
            split_id=bundle.manifest.split_id,
            protocol_variant="canonical_section_own",
            k=k,
            metric=metric,
            weighting=weighting.value,
            edge_index_convention="source_to_target",
            num_nodes=n_total,
            num_edges=edge_index.size(1),
            num_train_nodes=n_tr,
            num_val_nodes=n_va,
            num_test_nodes=n_te,
            num_train_train_edges=num_tr_tr,
            num_val_val_edges=num_va_va,
            num_test_test_edges=num_te_te,
            num_disallowed_cross_partition_edges=num_disallowed_part,
            num_disallowed_cross_section_edges=num_disallowed_sec,
            edge_index_hash=hash_array(edge_index.numpy()),
            edge_weight_hash=hash_array(edge_weight_tensor.numpy())
            if edge_weight_tensor is not None
            else None,
            feature_manifest_hash=bundle.manifest.compute_manifest_hash(),
        )

        return GraphBundle(
            edge_index=edge_index,
            num_nodes=n_total,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_cell_ids=cell_ids,
            manifest=manifest,
            edge_weight=edge_weight_tensor,
            node_section_ids=sections,
        )
