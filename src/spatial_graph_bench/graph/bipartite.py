"""Secondary Variant A: Bipartite reference connectivity graph builder."""

from __future__ import annotations

import torch
from sklearn.neighbors import NearestNeighbors

from spatial_graph_bench.graph.base import BaseGraphBuilder
from spatial_graph_bench.graph.schema import GraphBundle, GraphManifest
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.hashing import hash_array
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("graph.bipartite")


class BipartiteReferenceGraphBuilder(BaseGraphBuilder):
    """Constructs bipartite reference graphs (scgraph-bench v0 protocol).

    - Train -> Train: symmetrized k-NN on feature PCA.
    - Train -> Val: directed edges from train reference to val query.
    - Train -> Test: directed edges from train reference to test query.
    - Query -> Query: explicitly disabled (0 val-val, 0 test-test).
    """

    def __init__(self, k: int = 20, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric

    def build(
        self,
        bundle: PreprocessedBundle,
        graph_name: str,
        extra_metadata: dict | None = None,
    ) -> GraphBundle:
        _ = extra_metadata
        n_tr = len(bundle.train_cell_ids)
        n_va = len(bundle.val_cell_ids)
        n_te = len(bundle.test_cell_ids)
        n_total = n_tr + n_va + n_te

        logger.info("Building bipartite reference graph (Variant A, k=%d)...", self.k)

        # 1. Fit NN exclusively on train features
        nn_train = NearestNeighbors(n_neighbors=min(self.k + 1, n_tr), metric=self.metric)
        nn_train.fit(bundle.X_pca_train)

        # Train -> Train
        train_dists, train_indices = nn_train.kneighbors(bundle.X_pca_train)
        edges: set[tuple[int, int]] = set()

        eff_k_tr = min(self.k, n_tr - 1)
        for i in range(n_tr):
            for rank in range(1, eff_k_tr + 1):
                j = int(train_indices[i, rank])
                edges.add((j, i))
                edges.add((i, j))

        num_tr_tr = len(edges)

        # Train -> Val
        if n_va > 0:
            val_dists, val_indices = nn_train.kneighbors(
                bundle.X_pca_val, n_neighbors=min(self.k, n_tr)
            )
            for u in range(n_va):
                target_val = n_tr + u
                for rank in range(min(self.k, n_tr)):
                    source_train = int(val_indices[u, rank])
                    edges.add((source_train, target_val))

        # Train -> Test
        if n_te > 0:
            test_dists, test_indices = nn_train.kneighbors(
                bundle.X_pca_test, n_neighbors=min(self.k, n_tr)
            )
            for w in range(n_te):
                target_test = n_tr + n_va + w
                for rank in range(min(self.k, n_tr)):
                    source_train = int(test_indices[w, rank])
                    edges.add((source_train, target_test))

        sorted_edges = sorted(edges)
        edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()

        train_mask = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_tr] = True
        val_mask = torch.zeros(n_total, dtype=torch.bool)
        val_mask[n_tr : n_tr + n_va] = True
        test_mask = torch.zeros(n_total, dtype=torch.bool)
        test_mask[n_tr + n_va :] = True

        cell_ids = bundle.train_cell_ids + bundle.val_cell_ids + bundle.test_cell_ids
        sections = bundle.section_ids_train + bundle.section_ids_val + bundle.section_ids_test

        manifest = GraphManifest(
            graph_name=graph_name,
            builder_type="bipartite_reference",
            dataset_name=bundle.manifest.dataset_name,
            split_id=bundle.manifest.split_id,
            protocol_variant="variant_a_bipartite",
            k=self.k,
            metric=self.metric,
            weighting="unweighted",
            edge_index_convention="source_to_target",
            num_nodes=n_total,
            num_edges=edge_index.size(1),
            num_train_nodes=n_tr,
            num_val_nodes=n_va,
            num_test_nodes=n_te,
            num_train_train_edges=num_tr_tr,
            num_val_val_edges=0,
            num_test_test_edges=0,
            num_disallowed_cross_partition_edges=0,
            num_disallowed_cross_section_edges=0,
            edge_index_hash=hash_array(edge_index.numpy()),
            edge_weight_hash=None,
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
            edge_weight=None,
            node_section_ids=sections,
        )
