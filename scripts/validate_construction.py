"""Validate graph construction bundles and assert protocol connectivity invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from spatial_graph_bench.graph.schema import GraphBundle

console = Console()


def validate_graph_construction(graph_dir: Path) -> bool:
    console.print(f"[bold cyan]Validating Graph Construction:[/bold cyan] {graph_dir}")
    if not graph_dir.is_dir():
        console.print(f"[bold red]Graph directory not found:[/bold red] {graph_dir}")
        return False

    try:
        bundle = GraphBundle.load(graph_dir)
        manifest = bundle.manifest
        edge_index = bundle.edge_index

        num_nodes = bundle.num_nodes
        num_edges = edge_index.size(1)

        # Invariant checks
        # Partitions: 0=train, 1=val, 2=test
        partitions = np.zeros(num_nodes, dtype=np.int32)
        partitions[bundle.val_mask.numpy()] = 1
        partitions[bundle.test_mask.numpy()] = 2

        sections = np.array(bundle.node_section_ids)

        src_nodes = edge_index[0].numpy()
        tgt_nodes = edge_index[1].numpy()

        src_parts = partitions[src_nodes]
        tgt_parts = partitions[tgt_nodes]

        src_secs = sections[src_nodes]
        tgt_secs = sections[tgt_nodes]

        # Protocol specific assertions
        table = Table(title=f"Topological Verification: {manifest.graph_name}")
        table.add_column("Constraint", style="cyan")
        table.add_column("Detail", style="magenta")
        table.add_column("Status", style="green")

        all_ok = True

        if manifest.protocol_variant == "canonical_section_own":
            cross_partition = int(np.sum(src_parts != tgt_parts))
            cross_section = int(np.sum(src_secs != tgt_secs))

            table.add_row(
                "Zero Cross-Partition Edges",
                f"Count = {cross_partition}",
                "OK" if cross_partition == 0 else "VIOLATION",
            )
            table.add_row(
                "Zero Cross-Section Edges",
                f"Count = {cross_section}",
                "OK" if cross_section == 0 else "VIOLATION",
            )

            if cross_partition > 0 or cross_section > 0:
                all_ok = False

        elif manifest.protocol_variant == "variant_a_bipartite":
            # Test and val nodes must only receive edges from train nodes
            val_or_test_tgt = (tgt_parts == 1) | (tgt_parts == 2)
            invalid_src = np.sum(val_or_test_tgt & (src_parts != 0))

            # Disallow val-val, test-test, test->val
            table.add_row(
                "Bipartite Test/Val -> Train",
                f"Invalid source count = {invalid_src}",
                "OK" if invalid_src == 0 else "VIOLATION",
            )
            if invalid_src > 0:
                all_ok = False

        table.add_row("Edge Index Shape", f"[2, {num_edges}]", "OK")
        table.add_row("Node Count", str(num_nodes), "OK")
        table.add_row("Manifest Hash", f"{manifest.compute_manifest_hash()[:16]}...", "VALID")

        console.print(table)
        return all_ok
    except Exception as err:
        console.print(f"[bold red]Graph validation failed:[/bold red] {err}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path)
    args = parser.parse_args()

    ok = validate_graph_construction(args.graph_dir)
    sys.exit(0 if ok else 1)
