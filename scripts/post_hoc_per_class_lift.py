"""Post-Hoc Analysis: Per-Class Cell-Type Lift and Biological Dissection.

Quantifies per-class F1 score and matched lift relative to the non-spatial MLP
baseline across all 10 evaluation seeds, revealing which biological cell states
benefit from or are corrupted by spatial graph message passing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score

console = Console()


def analyze_per_class_lift(
    dataset_name: str = "stereoseq_axolotl_telencephalon",
    split_id: str = "developmental_three_stage",
    model: str = "graphsage",
    graph: str = "spatial_knn_k12",
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if seeds is None:
        seeds = list(range(42, 52))

    prep_dir = Path(f"artifacts/preprocessed/{dataset_name}/{split_id}")
    res_dir = Path(f"artifacts/results/{dataset_name}/{split_id}")
    snapshot_dir = Path(f"audits/baselines_snapshot/{dataset_name}/{split_id}")
    split_file = Path(f"splits/{dataset_name}/{split_id}.json")

    with open(prep_dir / "label_mapping.json") as f:
        label_to_id = json.load(f)

    with open(split_file) as f:
        split_meta = json.load(f)
    eval_labels = split_meta["evaluated_labels"]
    eval_ids = [label_to_id[lab] for lab in eval_labels if lab in label_to_id]

    y_true = np.load(prep_dir / "test_labels.npy")

    # 1. Compute MLP baseline per-class F1 across seeds
    mlp_f1_matrix = []
    for seed in seeds:
        mlp_run = res_dir / f"mlp_none_seed{seed}"
        if not mlp_run.is_dir():
            mlp_run = snapshot_dir / f"mlp_none_seed{seed}"

        metrics_file = mlp_run / "metrics_summary.json"
        preds_file = mlp_run / "test_preds.npy"
        if metrics_file.is_file():
            with open(metrics_file) as f:
                m_data = json.load(f)
            class_f1s = [m_data["test"]["per_class_f1"].get(lab, 0.0) for lab in eval_labels]
            mlp_f1_matrix.append(class_f1s)
        elif preds_file.is_file():
            preds = np.load(preds_file)
            f1s = f1_score(y_true, preds, labels=eval_ids, average=None, zero_division=0.0)
            mlp_f1_matrix.append(f1s)
        else:
            raise FileNotFoundError(
                f"Neither metrics_summary.json nor test_preds.npy found in {mlp_run}"
            )
    mlp_f1_matrix = np.array(mlp_f1_matrix)  # shape: (n_seeds, n_classes)

    # 2. Compute GNN per-class F1 across seeds
    gnn_f1_matrix = []
    for seed in seeds:
        run_path = res_dir / f"{model}_{graph}_seed{seed}"
        metrics_file = run_path / "metrics_summary.json"
        preds_file = run_path / "test_preds.npy"
        if metrics_file.is_file():
            with open(metrics_file) as f:
                m_data = json.load(f)
            class_f1s = [m_data["test"]["per_class_f1"].get(lab, 0.0) for lab in eval_labels]
            gnn_f1_matrix.append(class_f1s)
        elif preds_file.is_file():
            preds = np.load(preds_file)
            f1s = f1_score(y_true, preds, labels=eval_ids, average=None, zero_division=0.0)
            gnn_f1_matrix.append(f1s)
        else:
            raise FileNotFoundError(
                f"Neither metrics_summary.json nor test_preds.npy found in {run_path}"
            )
    gnn_f1_matrix = np.array(gnn_f1_matrix)

    lift_matrix = gnn_f1_matrix - mlp_f1_matrix

    mean_mlp = mlp_f1_matrix.mean(axis=0)
    std_mlp = mlp_f1_matrix.std(axis=0)
    mean_gnn = gnn_f1_matrix.mean(axis=0)
    std_gnn = gnn_f1_matrix.std(axis=0)
    mean_lift = lift_matrix.mean(axis=0)
    std_lift = lift_matrix.std(axis=0)

    class_results = []
    for i, (lab, lab_id) in enumerate(zip(eval_labels, eval_ids, strict=True)):
        class_count = int(np.sum(y_true == lab_id))
        class_results.append(
            {
                "class_name": lab,
                "class_id": lab_id,
                "support": class_count,
                "gnn_f1_mean": float(mean_gnn[i]),
                "gnn_f1_std": float(std_gnn[i]),
                "mlp_f1_mean": float(mean_mlp[i]),
                "mlp_f1_std": float(std_mlp[i]),
                "lift_mean": float(mean_lift[i]),
                "lift_std": float(std_lift[i]),
            }
        )

    # Sort descending by lift
    class_results.sort(key=lambda x: x["lift_mean"], reverse=True)

    summary = {
        "dataset_name": dataset_name,
        "split_id": split_id,
        "model": model,
        "graph": graph,
        "n_seeds": len(seeds),
        "overall_gnn_f1": float(mean_gnn.mean()),
        "overall_mlp_f1": float(mean_mlp.mean()),
        "overall_lift": float(mean_lift.mean()),
        "classes": class_results,
    }

    out_file = res_dir / f"per_class_lift_{model}_{graph}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Display Rich Table
    table = Table(
        title=f"Per-Class Lift Breakdown: {model.upper()} on {graph}\nDataset: {dataset_name} ({split_id})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Cell Type", style="bold cyan")
    table.add_column("Support", justify="right")
    table.add_column("GNN F1 (mean ± std)", justify="center")
    table.add_column("MLP Base (mean ± std)", justify="center")
    table.add_column("Matched Lift Δ", justify="center")
    table.add_column("Verdict", justify="center")

    for cr in class_results:
        lift = cr["lift_mean"]
        if lift > 0.02:
            verdict = "[bold green]POSITIVE[/bold green]"
            lift_str = f"[green]+{lift:.4f} ± {cr['lift_std']:.4f}[/green]"
        elif lift < -0.02:
            verdict = "[bold red]NEGATIVE[/bold red]"
            lift_str = f"[red]{lift:.4f} ± {cr['lift_std']:.4f}[/red]"
        else:
            verdict = "[yellow]PARITY[/yellow]"
            lift_str = f"{lift:+.4f} ± {cr['lift_std']:.4f}"

        table.add_row(
            cr["class_name"],
            f"{cr['support']:,}",
            f"{cr['gnn_f1_mean']:.4f} ± {cr['gnn_f1_std']:.4f}",
            f"{cr['mlp_f1_mean']:.4f} ± {cr['mlp_f1_std']:.4f}",
            lift_str,
            verdict,
        )

    console.print(table)
    console.print(f"[bold green]Saved summary JSON to:[/bold green] {out_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-class lift relative to MLP baseline.")
    parser.add_argument("--dataset", default="stereoseq_axolotl_telencephalon")
    parser.add_argument("--split", default="developmental_three_stage")
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--graph", default="spatial_knn_k12")
    args = parser.parse_args()

    analyze_per_class_lift(
        dataset_name=args.dataset,
        split_id=args.split,
        model=args.model,
        graph=args.graph,
    )


if __name__ == "__main__":
    main()
