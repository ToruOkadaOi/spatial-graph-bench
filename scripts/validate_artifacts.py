"""Validate all benchmark artifacts against cryptographic and topological invariants."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.paths import ArtifactPaths

console = Console()


def validate_all_artifacts(dataset_name: str, split_id: str) -> bool:
    paths = ArtifactPaths.default()
    console.print(
        f"[bold blue]Starting Artifact Validation for {dataset_name} ({split_id})...[/bold blue]\n"
    )

    all_passed = True
    table = Table(title="Benchmark Artifact Integrity & Validation Status")
    table.add_column("Category", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Details", style="yellow")
    table.add_column("Status", style="green")

    # 1. Split
    split_file = paths.dataset_split_file(dataset_name, split_id)
    if split_file.is_file():
        try:
            split = SplitDefinition.load_json(split_file)
            table.add_row(
                "Frozen Split",
                f"{split_id}.json",
                f"Train: {len(split.train_cell_ids)}, Val: {len(split.val_cell_ids)}, Test: {len(split.test_cell_ids)}",
                "[bold green]PASSED[/bold green]",
            )
        except Exception as e:
            table.add_row("Frozen Split", f"{split_id}.json", str(e), "[bold red]FAILED[/bold red]")
            all_passed = False
    else:
        table.add_row(
            "Frozen Split", f"{split_id}.json", "FILE MISSING", "[bold red]FAILED[/bold red]"
        )
        all_passed = False

    # 2. Features
    prep_dir = paths.dataset_preprocessed_dir(dataset_name, split_id)
    if (prep_dir / "feature_manifest.json").is_file():
        try:
            bundle = PreprocessedBundle.load(prep_dir)
            table.add_row(
                "Feature Bundle",
                "PCA Features",
                f"Train: {bundle.X_pca_train.shape}, Val: {bundle.X_pca_val.shape}, Test: {bundle.X_pca_test.shape}",
                "[bold green]PASSED[/bold green]",
            )
        except Exception as e:
            table.add_row("Feature Bundle", "PCA Features", str(e), "[bold red]FAILED[/bold red]")
            all_passed = False
    else:
        table.add_row(
            "Feature Bundle", "PCA Features", "DIR MISSING", "[bold red]FAILED[/bold red]"
        )
        all_passed = False

    # 3. Graphs
    graph_base = paths.graphs_dir / dataset_name / split_id
    if graph_base.is_dir():
        graphs = sorted(
            [
                d
                for d in graph_base.iterdir()
                if d.is_dir() and (d / "graph_manifest.json").is_file()
            ]
        )
        for g_dir in graphs:
            try:
                g_bundle = GraphBundle.load(g_dir)
                table.add_row(
                    "Graph Bundle",
                    g_dir.name,
                    f"Nodes: {g_bundle.num_nodes}, Edges: {g_bundle.edge_index.size(1)}, Variant: {g_bundle.manifest.protocol_variant}",
                    "[bold green]PASSED[/bold green]",
                )
            except Exception as e:
                table.add_row("Graph Bundle", g_dir.name, str(e), "[bold red]FAILED[/bold red]")
                all_passed = False

    console.print(table)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    ok = validate_all_artifacts(args.dataset, args.split)
    sys.exit(0 if ok else 1)
