"""Construct canonical spatial k-NN graphs, negative controls, and Variant A graphs."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from spatial_graph_bench.config.graph import (
    CoordinateShuffleConfig,
    RewiredControlConfig,
    SpatialkNNConfig,
)
from spatial_graph_bench.graph.audit import validate_graph_construction
from spatial_graph_bench.graph.bipartite import BipartiteReferenceGraphBuilder
from spatial_graph_bench.graph.coordinate_shuffle import build_coordinate_shuffled_graph
from spatial_graph_bench.graph.rewired_control import build_rewired_control_graph
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.build_spatial_graphs")
console = Console()


def build_graphs_for_split(
    dataset_name: str,
    split_id: str,
    k_values: list[int] = (6, 12),
    bipartite_k: int = 20,
    seed: int = 42,
) -> list[tuple[str, str, int, int, str, bool]]:
    paths = ArtifactPaths.default()
    prep_dir = paths.dataset_preprocessed_dir(dataset_name, split_id)

    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed bundle directory not found: {prep_dir}")

    logger.info("Loading preprocessed feature bundle from %s...", prep_dir)
    bundle = PreprocessedBundle.load(prep_dir)

    results: list[tuple[str, str, int, int, str, bool]] = []

    for k in k_values:
        # 1. Canonical spatial k-NN graph
        graph_name = f"spatial_knn_k{k}"
        logger.info("Building canonical spatial k-NN graph: %s...", graph_name)
        knn_builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=k))
        knn_bundle = knn_builder.build(bundle, graph_name=graph_name)

        out_dir = paths.dataset_graph_dir(dataset_name, split_id, graph_name)
        knn_bundle.save(out_dir)
        valid = validate_graph_construction(out_dir)
        results.append(
            (
                graph_name,
                knn_bundle.manifest.protocol_variant,
                knn_bundle.num_nodes,
                knn_bundle.manifest.num_edges,
                knn_bundle.manifest.compute_manifest_hash()[:16],
                valid,
            )
        )

        # 2. Degree-preserving rewired negative control
        rewired_name = f"rewired_spatial_knn_k{k}"
        logger.info("Building degree-preserving rewired control: %s...", rewired_name)
        rewired_cfg = RewiredControlConfig(
            reference_graph_name=graph_name,
            seed=seed,
        )
        rewired_bundle = build_rewired_control_graph(
            base_bundle=knn_bundle,
            config=rewired_cfg,
            graph_name=rewired_name,
        )
        out_rewired_dir = paths.dataset_graph_dir(dataset_name, split_id, rewired_name)
        rewired_bundle.save(out_rewired_dir)
        valid_rewired = validate_graph_construction(out_rewired_dir)
        results.append(
            (
                rewired_name,
                rewired_bundle.manifest.protocol_variant,
                rewired_bundle.num_nodes,
                rewired_bundle.manifest.num_edges,
                rewired_bundle.manifest.compute_manifest_hash()[:16],
                valid_rewired,
            )
        )

        # 3. Coordinate-shuffled negative control
        shuffled_name = f"shuffled_spatial_knn_k{k}"
        logger.info("Building coordinate-shuffled control: %s...", shuffled_name)
        shuffled_cfg = CoordinateShuffleConfig(
            reference_graph_name=graph_name,
            seed=seed,
        )
        shuffled_bundle = build_coordinate_shuffled_graph(
            bundle=bundle,
            config=shuffled_cfg,
            graph_name=shuffled_name,
            k=k,
        )
        out_shuffled_dir = paths.dataset_graph_dir(dataset_name, split_id, shuffled_name)
        shuffled_bundle.save(out_shuffled_dir)
        valid_shuffled = validate_graph_construction(out_shuffled_dir)
        results.append(
            (
                shuffled_name,
                shuffled_bundle.manifest.protocol_variant,
                shuffled_bundle.num_nodes,
                shuffled_bundle.manifest.num_edges,
                shuffled_bundle.manifest.compute_manifest_hash()[:16],
                valid_shuffled,
            )
        )

    # 4. Variant A: Bipartite Reference Graph
    bipartite_name = f"bipartite_ref_k{bipartite_k}"
    logger.info("Building Variant A bipartite reference graph: %s...", bipartite_name)
    bipartite_builder = BipartiteReferenceGraphBuilder(k=bipartite_k)
    bipartite_bundle = bipartite_builder.build(bundle, graph_name=bipartite_name)
    out_bipartite_dir = paths.dataset_graph_dir(dataset_name, split_id, bipartite_name)
    bipartite_bundle.save(out_bipartite_dir)
    valid_bipartite = validate_graph_construction(out_bipartite_dir)
    results.append(
        (
            bipartite_name,
            bipartite_bundle.manifest.protocol_variant,
            bipartite_bundle.num_nodes,
            bipartite_bundle.manifest.num_edges,
            bipartite_bundle.manifest.compute_manifest_hash()[:16],
            valid_bipartite,
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument(
        "--split",
        type=str,
        default="mouse_held_out_canonical",
        help="Split ID to build graphs for (or 'all').",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--bipartite-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ArtifactPaths.default()
    if args.split == "all":
        dataset_prep = paths.preprocessed_dir / args.dataset
        split_ids = [p.name for p in dataset_prep.iterdir() if p.is_dir()]
    else:
        split_ids = [args.split]

    all_passed = True
    for split_id in split_ids:
        console.print(
            f"\n[bold green]>>> Building Graph Constructions for {args.dataset} / {split_id}[/bold green]"
        )
        results = build_graphs_for_split(
            dataset_name=args.dataset,
            split_id=split_id,
            k_values=args.k,
            bipartite_k=args.bipartite_k,
            seed=args.seed,
        )

        table = Table(title=f"Graph Constructions Summary: {args.dataset} ({split_id})")
        table.add_column("Graph Name", style="cyan")
        table.add_column("Protocol Variant", style="magenta")
        table.add_column("Nodes", justify="right")
        table.add_column("Edges", justify="right")
        table.add_column("Manifest Hash", style="dim")
        table.add_column("Status", style="bold")

        for name, variant, n_nodes, n_edges, h, ok in results:
            table.add_row(
                name,
                variant,
                f"{n_nodes:,}",
                f"{n_edges:,}",
                f"{h}...",
                "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            )
            if not ok:
                all_passed = False

        console.print(table)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
