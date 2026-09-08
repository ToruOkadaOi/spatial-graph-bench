"""Summarize and inspect benchmark sweep results.

Usage:
    uv run python scripts/summarize_results.py
    uv run python scripts/summarize_results.py --per-class
    uv run python scripts/summarize_results.py --run gcn_spatial_knn_k6_seed42
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


def load_all_results(results_dir: Path):
    # 1. Load MLP baselines
    mlp_baselines = {}
    for seed in range(42, 52):
        mlp_f = results_dir / f"mlp_none_seed{seed}" / "metrics_summary.json"
        if not mlp_f.exists():
            mlp_f = Path("artifacts/snapshots/merfish_mouse_spinal_cord/mouse_held_out_canonical") / f"mlp_none_seed{seed}" / "metrics_summary.json"
        if mlp_f.exists():
            with open(mlp_f) as f:
                m = json.load(f)
            mlp_baselines[seed] = m["test"]["macro_f1"]

    data = defaultdict(lambda: {"gnn": [], "mlp": [], "lift": [], "time": [], "epoch": []})
    models = ["gcn", "gat", "gin", "graphsage"]
    graphs = [
        "spatial_knn_k6", "spatial_knn_k12",
        "rewired_spatial_knn_k6", "rewired_spatial_knn_k12",
        "shuffled_spatial_knn_k6", "shuffled_spatial_knn_k12",
        "bipartite_ref_k20",
    ]

    for mod in models:
        for gr in graphs:
            for seed in range(42, 52):
                run_id = f"{mod}_{gr}_seed{seed}"
                run_dir = results_dir / run_id
                m_path = run_dir / "metrics_summary.json"
                man_path = run_dir / "run_manifest.json"
                if m_path.exists():
                    with open(m_path) as f:
                        m = json.load(f)
                    gnn_f1 = m["test"]["macro_f1"]
                    mlp_f1 = mlp_baselines.get(seed, 0.5273)
                    lift = gnn_f1 - mlp_f1
                    data[(mod, gr)]["gnn"].append(gnn_f1)
                    data[(mod, gr)]["mlp"].append(mlp_f1)
                    data[(mod, gr)]["lift"].append(lift)

                    if man_path.exists():
                        with open(man_path) as f:
                            man = json.load(f)
                        data[(mod, gr)]["time"].append(man.get("training_time_seconds", 0))
                        data[(mod, gr)]["epoch"].append(man.get("best_epoch", 0))

    return data, models, graphs


def print_summary_table(data, models, graphs, epsilon=0.0069):
    table = Table(title="GNN Benchmark Sweep: 280 Runs across 10 Seeds (merfish_mouse_spinal_cord)")
    table.add_column("Model", style="cyan", justify="left")
    table.add_column("Graph Construction", style="magenta", justify="left")
    table.add_column("N", justify="center")
    table.add_column("GNN Test F1", justify="right")
    table.add_column("MLP Base F1", justify="right")
    table.add_column("Matched Lift (Δ)", justify="right", style="bold")
    table.add_column("Avg Time", justify="right")
    table.add_column("Classification", justify="center", style="bold")

    for mod in models:
        for gr in graphs:
            v = data[(mod, gr)]
            n = len(v["gnn"])
            if n == 0:
                continue
            g_mean, g_std = np.mean(v["gnn"]), np.std(v["gnn"])
            m_mean, m_std = np.mean(v["mlp"]), np.std(v["mlp"])
            l_mean, l_std = np.mean(v["lift"]), np.std(v["lift"])
            t_mean = np.mean(v["time"]) if v["time"] else 0

            if l_mean > epsilon:
                verdict = "[green]POSITIVE[/green]"
            elif l_mean < -epsilon:
                verdict = "[red]NEGATIVE[/red]"
            else:
                verdict = "[yellow]PARITY[/yellow]"

            table.add_row(
                mod.upper(),
                gr,
                str(n),
                f"{g_mean:.4f} ± {g_std:.4f}",
                f"{m_mean:.4f} ± {m_std:.4f}",
                f"{l_mean:+.4f} ± {l_std:.4f}",
                f"{t_mean:.1f}s",
                verdict,
            )

    console.print(table)


def print_per_class(results_dir: Path):
    mlp_classes = defaultdict(list)
    for s in range(42, 52):
        p = results_dir / f"mlp_none_seed{s}" / "metrics_summary.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            for c, score in d["test"]["per_class_f1"].items():
                mlp_classes[c].append(score)

    mlp_mean = {c: np.mean(v) for c, v in mlp_classes.items()}

    gcn_classes = defaultdict(list)
    gin_bip_classes = defaultdict(list)
    for s in range(42, 52):
        p1 = results_dir / f"gcn_spatial_knn_k6_seed{s}" / "metrics_summary.json"
        if p1.exists():
            with open(p1) as f:
                d = json.load(f)
            for c, score in d["test"]["per_class_f1"].items():
                gcn_classes[c].append(score)

        p2 = results_dir / f"gin_bipartite_ref_k20_seed{s}" / "metrics_summary.json"
        if p2.exists():
            with open(p2) as f:
                d = json.load(f)
            for c, score in d["test"]["per_class_f1"].items():
                gin_bip_classes[c].append(score)

    records = []
    for c in mlp_mean:
        records.append({
            "class": c,
            "mlp": mlp_mean[c],
            "gcn": np.mean(gcn_classes[c]) if gcn_classes[c] else 0,
            "gcn_delta": (np.mean(gcn_classes[c]) if gcn_classes[c] else 0) - mlp_mean[c],
            "gin_bip": np.mean(gin_bip_classes[c]) if gin_bip_classes[c] else 0,
            "gin_delta": (np.mean(gin_bip_classes[c]) if gin_bip_classes[c] else 0) - mlp_mean[c],
        })

    # Top collapsed
    table1 = Table(title="Top 10 Cell Types with Worst Collapse in GCN (spatial_knn_k6)")
    table1.add_column("Cell Type", style="cyan")
    table1.add_column("MLP F1", justify="right")
    table1.add_column("GCN F1", justify="right")
    table1.add_column("Δ Collapse", justify="right", style="bold red")

    for r in sorted(records, key=lambda x: x["gcn_delta"])[:10]:
        table1.add_row(r["class"], f"{r['mlp']:.4f}", f"{r['gcn']:.4f}", f"{r['gcn_delta']:+.4f}")
    console.print(table1)

    # Top gained
    table2 = Table(title="Top 10 Cell Types with Highest Gain in GIN (bipartite_ref_k20)")
    table2.add_column("Cell Type", style="cyan")
    table2.add_column("MLP F1", justify="right")
    table2.add_column("GIN Bip F1", justify="right")
    table2.add_column("Δ Gain", justify="right", style="bold green")

    for r in sorted(records, key=lambda x: x["gin_delta"], reverse=True)[:10]:
        table2.add_row(r["class"], f"{r['mlp']:.4f}", f"{r['gin_bip']:.4f}", f"{r['gin_delta']:+.4f}")
    console.print(table2)


def inspect_run(run_dir: Path):
    if not run_dir.exists():
        console.print(f"[red]Directory does not exist: {run_dir}[/red]")
        return

    m_path = run_dir / "metrics_summary.json"
    man_path = run_dir / "run_manifest.json"

    if man_path.exists():
        with open(man_path) as f:
            man = json.load(f)
        console.print(f"[bold green]Run Manifest: {man.get('run_id')}[/bold green]")
        console.print(f"  Model: {man.get('model_name')} | Seed: {man.get('seed')}")
        console.print(f"  Best Epoch: {man.get('best_epoch')} | Training Time: {man.get('training_time_seconds', 0):.2f}s")
        console.print(f"  Best Val Macro-F1: {man.get('best_val_macro_f1'):.4f}")

    if m_path.exists():
        with open(m_path) as f:
            m = json.load(f)
        console.print("\n[bold]Test Metrics:[/bold]")
        console.print(f"  Test Macro-F1: [bold cyan]{m['test']['macro_f1']:.4f}[/bold cyan]")
        console.print(f"  Test Balanced Accuracy: {m['test']['balanced_accuracy']:.4f}")
        console.print(f"  Per-Section F1: {m['test'].get('per_section_macro_f1', {})}")


def main():
    parser = argparse.ArgumentParser(description="Inspect benchmark sweep results.")
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results/merfish_mouse_spinal_cord/mouse_held_out_canonical"))
    parser.add_argument("--per-class", action="store_true", help="Show per-class top collapse and gains")
    parser.add_argument("--run", type=str, help="Inspect a specific run folder name")
    args = parser.parse_args()

    if args.run:
        inspect_run(args.results_dir / args.run)
    elif args.per_class:
        print_per_class(args.results_dir)
    else:
        data, models, graphs = load_all_results(args.results_dir)
        print_summary_table(data, models, graphs)


if __name__ == "__main__":
    main()
