"""Execute GNN sweeps across architectures, graph constructions, and controls.

Covers the pre-registered co-equal GNN architecture set: {GCN, GraphSAGE, GAT, GIN}
across {spatial_knn, rewired_control, shuffled_control, bipartite_ref}.
Computes matched lift against same-seed frozen MLP baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from spatial_graph_bench.analysis.delivery import AuditVerdict, audit_run_dir
from spatial_graph_bench.config.model import (
    BenchmarkRunConfig,
    GNNConfig,
    ModelType,
    TrainingConfig,
)
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.models.trainer import run_benchmark_training
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.tracking.graph_lift import compute_matched_graph_lift
from spatial_graph_bench.tracking.schema import RunManifest
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.run_gnn_sweep")
console = Console()


def run_gnn_sweep(
    dataset_name: str,
    split_id: str,
    models: list[str],
    graphs: list[str],
    seeds: list[int],
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    parity_band_halfwidth: float = 0.0069,
) -> list[dict[str, Any]]:
    paths = ArtifactPaths.default()
    prep_dir = paths.dataset_preprocessed_dir(dataset_name, split_id)
    results_dir = paths.dataset_results_dir(dataset_name, split_id)
    snapshot_dir = paths.audits_dir / "baselines_snapshot" / dataset_name / split_id

    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed bundle directory not found: {prep_dir}")

    # Load precomputed parity band if exists
    parity_file = snapshot_dir / "parity_band.json"
    if parity_file.is_file():
        pdata = json.loads(parity_file.read_text(encoding="utf-8"))
        parity_band_halfwidth = pdata.get("parity_band_halfwidth", parity_band_halfwidth)

    logger.info("Loading preprocessed feature bundle from %s...", prep_dir)
    feature_bundle = PreprocessedBundle.load(prep_dir)

    # Cache graph bundles
    loaded_graphs: dict[str, GraphBundle] = {}
    for g_name in graphs:
        g_dir = paths.dataset_graph_dir(dataset_name, split_id, g_name)
        if not g_dir.is_dir():
            raise FileNotFoundError(f"Graph directory not found: {g_dir}")
        logger.info("Loading graph bundle: %s...", g_name)
        loaded_graphs[g_name] = GraphBundle.load(g_dir)

    results: list[dict[str, Any]] = []

    console.print(
        f"\n[bold green]>>> Starting GNN Benchmark Sweep on {dataset_name}/{split_id}[/bold green]"
    )
    console.print(
        f"  Models: {models} | Graphs: {graphs} | Seeds: {seeds} | Parity Band: ±{parity_band_halfwidth:.4f}"
    )

    for m_str in models:
        model_type = ModelType(m_str.lower())
        for g_name in graphs:
            graph_bundle = loaded_graphs[g_name]
            for seed in seeds:
                logger.info("Running %s on %s (seed=%d)...", m_str.upper(), g_name, seed)

                # Match against frozen MLP baseline
                mlp_run_id = f"mlp_none_seed{seed}"
                mlp_manifest_file = results_dir / mlp_run_id / "run_manifest.json"
                mlp_metrics_file = results_dir / mlp_run_id / "metrics_summary.json"

                if not mlp_manifest_file.is_file():
                    # Check snapshot dir
                    mlp_manifest_file = snapshot_dir / mlp_run_id / "run_manifest.json"
                    mlp_metrics_file = snapshot_dir / mlp_run_id / "metrics_summary.json"

                if not mlp_manifest_file.is_file():
                    raise FileNotFoundError(
                        f"Baseline MLP run manifest missing for seed {seed}: {mlp_manifest_file}"
                    )

                mlp_manifest = RunManifest.model_validate_json(
                    mlp_manifest_file.read_text(encoding="utf-8")
                )
                mlp_metrics = json.loads(mlp_metrics_file.read_text(encoding="utf-8"))
                mlp_test_f1 = mlp_metrics["test"]["macro_f1"]
                mlp_test_bacc = mlp_metrics["test"]["balanced_accuracy"]

                gnn_cfg = BenchmarkRunConfig(
                    model_type=model_type,
                    dataset_name=dataset_name,
                    split_id=split_id,
                    graph_name=g_name,
                    gnn_config=GNNConfig(
                        hidden_dim=128,
                        num_layers=2,
                        dropout=0.2,
                    ),
                    training=TrainingConfig(
                        max_epochs=max_epochs,
                        learning_rate=lr,
                        weight_decay=weight_decay,
                        patience=patience,
                        seed=seed,
                        device=device,
                    ),
                )

                gnn_manifest, gnn_summaries = run_benchmark_training(
                    gnn_cfg,
                    feature_bundle=feature_bundle,
                    graph_bundle=graph_bundle,
                    out_dir=results_dir,
                )

                # 4-layer audit
                run_dir = results_dir / gnn_manifest.run_id
                audit_rep = audit_run_dir(
                    run_dir,
                    prep_bundle=feature_bundle,
                    graphs_root=paths.graphs_dir / dataset_name / split_id,
                )
                if audit_rep.verdict == AuditVerdict.FAIL:
                    raise RuntimeError(
                        f"Audit FAIL on {gnn_manifest.run_id}: {audit_rep.failed_hard_checks}"
                    )

                # Matched graph lift
                lift_rec = compute_matched_graph_lift(
                    gnn_manifest=gnn_manifest,
                    mlp_manifest=mlp_manifest,
                    gnn_macro_f1=gnn_summaries["test"]["macro_f1"],
                    mlp_macro_f1=mlp_test_f1,
                    gnn_balanced_acc=gnn_summaries["test"]["balanced_accuracy"],
                    mlp_balanced_acc=mlp_test_bacc,
                    parity_band_halfwidth=parity_band_halfwidth,
                    interior_lift=None,
                    boundary_lift=None,
                    per_section_lifts=None,
                    per_class_lifts=None,
                )

                results.append(
                    {
                        "model": m_str.upper(),
                        "graph": g_name,
                        "seed": seed,
                        "run_id": gnn_manifest.run_id,
                        "val_macro_f1": gnn_summaries["val"]["macro_f1"],
                        "test_macro_f1": gnn_summaries["test"]["macro_f1"],
                        "test_balanced_acc": gnn_summaries["test"]["balanced_accuracy"],
                        "mlp_matched_f1": mlp_test_f1,
                        "overall_lift": lift_rec.overall_graph_lift,
                        "parity_classification": lift_rec.parity_classification,
                        "best_epoch": gnn_manifest.best_epoch,
                        "training_time_seconds": gnn_manifest.training_time_seconds,
                    }
                )

                lift_str = (
                    f"{lift_rec.overall_graph_lift:+.4f} ({lift_rec.parity_classification.upper()})"
                )
                console.print(
                    f"  [cyan]{m_str.upper()}[/cyan] on [magenta]{g_name}[/magenta] (seed {seed}): "
                    f"Test Macro-F1 = [bold]{gnn_summaries['test']['macro_f1']:.4f}[/bold], "
                    f"MLP Baseline = {mlp_test_f1:.4f}, "
                    f"Lift = [bold]{lift_str}[/bold]"
                )

    # Summary table
    table = Table(title=f"GNN Benchmark Sweep & Matched Lift: {dataset_name} ({split_id})")
    table.add_column("Model", style="cyan")
    table.add_column("Graph Construction", style="magenta")
    table.add_column("Seed", justify="center")
    table.add_column("GNN Test F1", justify="right")
    table.add_column("MLP Base F1", justify="right")
    table.add_column("Matched Lift (Δ)", justify="right", style="bold")
    table.add_column("Classification", justify="center", style="bold")

    for r in results:
        p_class = r["parity_classification"]
        if p_class == "positive":
            color = "green"
        elif p_class == "negative":
            color = "red"
        else:
            color = "yellow"

        table.add_row(
            r["model"],
            r["graph"],
            str(r["seed"]),
            f"{r['test_macro_f1']:.4f}",
            f"{r['mlp_matched_f1']:.4f}",
            f"{r['overall_lift']:+.4f}",
            f"[{color}]{p_class.upper()}[/{color}]",
        )

    console.print(table)
    return results


def emit_gpu_batch_config(
    dataset_name: str,
    split_id: str,
    models: list[str],
    graphs: list[str],
    seeds: list[int],
    out_yaml: Path,
) -> None:
    """Generate reproducible YAML batch config for GPU worker execution."""
    cfg = {
        "batch_id": f"{dataset_name}_{split_id}_gnn_sweep",
        "dataset_name": dataset_name,
        "split_id": split_id,
        "models": models,
        "graphs": graphs,
        "seeds": seeds,
        "training": {
            "max_epochs": 200,
            "patience": 15,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cuda:0",
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    console.print(f"[bold green]GPU batch config generated:[/bold green] {out_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", type=str, default="mouse_held_out_canonical")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gcn", "graphsage", "gat", "gin"],
    )
    parser.add_argument(
        "--graphs",
        type=str,
        nargs="+",
        default=[
            "spatial_knn_k6",
            "rewired_spatial_knn_k6",
            "shuffled_spatial_knn_k6",
            "spatial_knn_k12",
            "rewired_spatial_knn_k12",
            "shuffled_spatial_knn_k12",
            "bipartite_ref_k20",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-config", type=Path, default=None)
    parser.add_argument("--emit-config", type=Path, default=None)
    args = parser.parse_args()

    paths = ArtifactPaths.default()
    if args.batch_config:
        cfg = yaml.safe_load(args.batch_config.read_text(encoding="utf-8"))
        dataset_name = cfg["dataset_name"]
        split_id = cfg["split_id"]
        models = cfg["models"]
        graphs = cfg["graphs"]
        seeds = cfg["seeds"]
        training_dict = cfg.get("training", {})
        max_epochs = training_dict.get("max_epochs", args.epochs)
        patience = training_dict.get("patience", args.patience)
        lr = training_dict.get("learning_rate", 1e-3)
        weight_decay = training_dict.get("weight_decay", 1e-4)
        device = training_dict.get("device", args.device)

        run_gnn_sweep(
            dataset_name=dataset_name,
            split_id=split_id,
            models=models,
            graphs=graphs,
            seeds=seeds,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        return

    if args.emit_config:
        emit_gpu_batch_config(
            dataset_name=args.dataset,
            split_id=args.split,
            models=args.models,
            graphs=args.graphs,
            seeds=args.seeds,
            out_yaml=args.emit_config,
        )
        return

    # Always ensure batch config exists in configs/
    default_cfg_path = paths.configs_dir / f"gpu_batch_{args.dataset}_{args.split}.yaml"
    emit_gpu_batch_config(
        dataset_name=args.dataset,
        split_id=args.split,
        models=args.models,
        graphs=args.graphs,
        seeds=args.seeds,
        out_yaml=default_cfg_path,
    )

    run_gnn_sweep(
        dataset_name=args.dataset,
        split_id=args.split,
        models=args.models,
        graphs=args.graphs,
        seeds=args.seeds,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
