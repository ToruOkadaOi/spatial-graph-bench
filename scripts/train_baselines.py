"""Train canonical spatially ignorant baselines (MLP >= 10 seeds, Random Forest).

Computes empirical sample standard deviation sigma_MLP, establishes the pre-registered
parity band (+/- 2*sigma_MLP), and freezes baseline snapshots in audits/baselines_snapshot/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from spatial_graph_bench.analysis.delivery import AuditVerdict, audit_run_dir
from spatial_graph_bench.config.model import (
    BenchmarkRunConfig,
    MLPConfig,
    ModelType,
    RandomForestConfig,
    TrainingConfig,
)
from spatial_graph_bench.models.trainer import run_benchmark_training
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.train_baselines")
console = Console()


def train_baselines_for_split(
    dataset_name: str,
    split_id: str,
    seeds: list[int] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    rf_estimators: int = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    paths = ArtifactPaths.default()
    prep_dir = paths.dataset_preprocessed_dir(dataset_name, split_id)
    results_dir = paths.dataset_results_dir(dataset_name, split_id)
    snapshot_dir = paths.audits_dir / "baselines_snapshot" / dataset_name / split_id

    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed bundle directory not found: {prep_dir}")

    logger.info("Loading preprocessed feature bundle from %s...", prep_dir)
    feature_bundle = PreprocessedBundle.load(prep_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    mlp_records: list[dict[str, Any]] = []

    console.print(
        f"\n[bold green]>>> Training MLP Baseline across {len(seeds)} seeds on {dataset_name}/{split_id}[/bold green]"
    )

    for seed in seeds:
        logger.info("Training MLP baseline (seed=%d)...", seed)
        mlp_cfg = BenchmarkRunConfig(
            model_type=ModelType.MLP,
            dataset_name=dataset_name,
            split_id=split_id,
            graph_name="none",
            mlp_config=MLPConfig(
                hidden_dims=[256, 128],
                dropout=0.2,
                use_batch_norm=True,
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

        manifest, summaries = run_benchmark_training(
            mlp_cfg,
            feature_bundle=feature_bundle,
            out_dir=results_dir,
        )

        # Audit run directory
        run_dir = results_dir / manifest.run_id
        audit_rep = audit_run_dir(run_dir, prep_bundle=feature_bundle)
        if audit_rep.verdict == AuditVerdict.FAIL:
            raise RuntimeError(f"Audit FAIL on MLP seed {seed}: {audit_rep.failed_hard_checks}")

        # Snapshot run manifest and metrics
        snap_run_dir = snapshot_dir / manifest.run_id
        snap_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "run_manifest.json", snap_run_dir / "run_manifest.json")
        shutil.copy(run_dir / "metrics_summary.json", snap_run_dir / "metrics_summary.json")

        mlp_records.append(
            {
                "seed": seed,
                "run_id": manifest.run_id,
                "train_macro_f1": summaries["train"]["macro_f1"],
                "val_macro_f1": summaries["val"]["macro_f1"],
                "test_macro_f1": summaries["test"]["macro_f1"],
                "test_balanced_acc": summaries["test"]["balanced_accuracy"],
                "label_coverage": summaries["test"]["label_coverage"],
                "best_epoch": manifest.best_epoch,
                "training_time_seconds": manifest.training_time_seconds,
            }
        )

        console.print(
            f"  [cyan]MLP (seed={seed})[/cyan]: Val F1 = {summaries['val']['macro_f1']:.4f}, "
            f"Test Macro-F1 = [bold]{summaries['test']['macro_f1']:.4f}[/bold], "
            f"Best Epoch = {manifest.best_epoch} ({manifest.training_time_seconds:.1f}s)"
        )

    # Compute MLP aggregate statistics
    test_f1s = np.array([r["test_macro_f1"] for r in mlp_records], dtype=np.float64)
    test_baccs = np.array([r["test_balanced_acc"] for r in mlp_records], dtype=np.float64)

    mean_f1 = float(np.mean(test_f1s))
    sigma_mlp = float(np.std(test_f1s, ddof=1)) if len(test_f1s) > 1 else 0.0
    parity_band_halfwidth = float(2.0 * sigma_mlp)
    tost_epsilon = max(0.005, parity_band_halfwidth)

    console.print(
        f"\n[bold green]>>> Training Random Forest Baseline on {dataset_name}/{split_id}[/bold green]"
    )
    rf_cfg = BenchmarkRunConfig(
        model_type=ModelType.RANDOM_FOREST,
        dataset_name=dataset_name,
        split_id=split_id,
        graph_name="none",
        rf_config=RandomForestConfig(
            n_estimators=rf_estimators,
            max_depth=30,
            min_samples_split=5,
            n_jobs=-1,
        ),
        training=TrainingConfig(seed=42),
    )

    rf_manifest, rf_summaries = run_benchmark_training(
        rf_cfg,
        feature_bundle=feature_bundle,
        out_dir=results_dir,
    )

    rf_run_dir = results_dir / rf_manifest.run_id
    rf_audit = audit_run_dir(rf_run_dir, prep_bundle=feature_bundle)
    if rf_audit.verdict == AuditVerdict.FAIL:
        raise RuntimeError(f"Audit FAIL on Random Forest: {rf_audit.failed_hard_checks}")

    snap_rf_dir = snapshot_dir / rf_manifest.run_id
    snap_rf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(rf_run_dir / "run_manifest.json", snap_rf_dir / "run_manifest.json")
    shutil.copy(rf_run_dir / "metrics_summary.json", snap_rf_dir / "metrics_summary.json")

    rf_record = {
        "run_id": rf_manifest.run_id,
        "seed": 42,
        "train_macro_f1": rf_summaries["train"]["macro_f1"],
        "val_macro_f1": rf_summaries["val"]["macro_f1"],
        "test_macro_f1": rf_summaries["test"]["macro_f1"],
        "test_balanced_acc": rf_summaries["test"]["balanced_accuracy"],
        "label_coverage": rf_summaries["test"]["label_coverage"],
        "training_time_seconds": rf_manifest.training_time_seconds,
    }

    console.print(
        f"  [cyan]Random Forest[/cyan]: Val F1 = {rf_summaries['val']['macro_f1']:.4f}, "
        f"Test Macro-F1 = [bold]{rf_summaries['test']['macro_f1']:.4f}[/bold] "
        f"({rf_manifest.training_time_seconds:.1f}s)"
    )

    # Parity Band summary object
    parity_band_info = {
        "dataset_name": dataset_name,
        "split_id": split_id,
        "num_mlp_seeds": len(seeds),
        "mlp_mean_test_macro_f1": mean_f1,
        "mlp_sigma_test_macro_f1": sigma_mlp,
        "parity_band_halfwidth": parity_band_halfwidth,
        "parity_band_interval": [mean_f1 - parity_band_halfwidth, mean_f1 + parity_band_halfwidth],
        "tost_equivalence_margin_epsilon": tost_epsilon,
        "rf_test_macro_f1": rf_summaries["test"]["macro_f1"],
    }

    full_summary = {
        "parity_band": parity_band_info,
        "mlp_seeds": mlp_records,
        "random_forest": rf_record,
        "aggregate_statistics": {
            "mlp_test_macro_f1_mean": mean_f1,
            "mlp_test_macro_f1_std": sigma_mlp,
            "mlp_test_macro_f1_min": float(np.min(test_f1s)),
            "mlp_test_macro_f1_max": float(np.max(test_f1s)),
            "mlp_test_balanced_acc_mean": float(np.mean(test_baccs)),
            "mlp_test_balanced_acc_std": (
                float(np.std(test_baccs, ddof=1)) if len(test_baccs) > 1 else 0.0
            ),
        },
    }

    # Save to audits/baselines_snapshot/
    (snapshot_dir / "baselines_summary.json").write_text(
        json.dumps(full_summary, indent=2), encoding="utf-8"
    )
    (snapshot_dir / "parity_band.json").write_text(
        json.dumps(parity_band_info, indent=2), encoding="utf-8"
    )

    # Summary table
    table = Table(title=f"Canonical Baselines & Parity Band: {dataset_name} ({split_id})")
    table.add_column("Model", style="cyan")
    table.add_column("Seed / Metric", style="magenta")
    table.add_column("Val Macro-F1", justify="right")
    table.add_column("Test Macro-F1", justify="right", style="bold")
    table.add_column("Test Bal Acc", justify="right")

    for r in mlp_records:
        table.add_row(
            "MLP",
            f"Seed {r['seed']}",
            f"{r['val_macro_f1']:.4f}",
            f"{r['test_macro_f1']:.4f}",
            f"{r['test_balanced_acc']:.4f}",
        )

    table.add_section()
    table.add_row(
        "[bold]MLP Mean ± Std[/bold]",
        f"across {len(seeds)} seeds",
        "-",
        f"[bold]{mean_f1:.4f} ± {sigma_mlp:.4f}[/bold]",
        f"{float(np.mean(test_baccs)):.4f} ± {float(np.std(test_baccs, ddof=1)):.4f}",
    )
    table.add_row(
        "[bold yellow]Parity Band (±2σ)[/bold yellow]",
        f"halfwidth = {parity_band_halfwidth:.4f}",
        "-",
        f"[{mean_f1 - parity_band_halfwidth:.4f}, {mean_f1 + parity_band_halfwidth:.4f}]",
        "-",
    )
    table.add_row(
        "[bold green]Random Forest[/bold green]",
        "Seed 42 (n=200)",
        f"{rf_record['val_macro_f1']:.4f}",
        f"[bold]{rf_record['test_macro_f1']:.4f}[/bold]",
        f"{rf_record['test_balanced_acc']:.4f}",
    )

    console.print(table)
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", type=str, default="mouse_held_out_canonical")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_baselines_for_split(
        dataset_name=args.dataset,
        split_id=args.split,
        seeds=args.seeds,
        max_epochs=args.epochs,
        patience=args.patience,
        rf_estimators=args.rf_estimators,
        device=args.device,
    )


if __name__ == "__main__":
    main()
