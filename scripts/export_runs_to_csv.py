"""Export granular per-run benchmark metrics and audit manifests to a single consolidated CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.export_runs_to_csv")


def export_runs_to_csv(
    dataset_name: str,
    split_id: str,
    output_csv: Path | None = None,
) -> Path:
    paths = ArtifactPaths.default()
    results_dir = paths.dataset_results_dir(dataset_name, split_id)
    baseline_dir = paths.audits_dir / "baselines_snapshot" / dataset_name / split_id

    if output_csv is None:
        reports_dir = paths.root_dir / "results" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_csv = reports_dir / f"{dataset_name}_{split_id}_all_runs_summary.csv"

    rows: list[dict[str, str | int | float]] = []

    # 1. Collect Baseline Runs
    if baseline_dir.is_dir():
        for run_dir in sorted(baseline_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name in (
                "baselines_summary.json",
                "parity_band.json",
            ):
                continue
            metrics_path = run_dir / "metrics_summary.json"
            manifest_path = run_dir / "run_manifest.json"
            if not metrics_path.is_file() or not manifest_path.is_file():
                continue

            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            model = "random_forest" if "random_forest" in run_dir.name else "mlp"
            parts = run_dir.name.split("_seed")
            seed = int(manifest.get("seed", parts[1] if len(parts) > 1 else 42))

            rows.append(
                {
                    "run_id": manifest.get("run_id", run_dir.name),
                    "run_type": "baseline",
                    "model": model,
                    "topology": "none",
                    "seed": seed,
                    "val_macro_f1": float(metrics.get("val", {}).get("macro_f1", 0.0)),
                    "test_macro_f1": float(metrics.get("test", {}).get("macro_f1", 0.0)),
                    "test_balanced_acc": float(
                        metrics.get("test", {}).get("balanced_accuracy", 0.0)
                    ),
                    "best_epoch": int(metrics.get("best_epoch", 1)),
                    "duration_seconds": float(
                        metrics.get(
                            "duration_seconds", manifest.get("training_time_seconds", 0.0) or 0.0
                        )
                    ),
                    "manifest_hash": str(
                        manifest.get("model_config_hash", manifest.get("run_hash", ""))
                    ),
                }
            )

    # 2. Collect GNN Sweep Runs
    if results_dir.is_dir():
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics_summary.json"
            manifest_path = run_dir / "run_manifest.json"
            if not metrics_path.is_file() or not manifest_path.is_file():
                continue

            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            # Skip baseline duplicates in results_dir if already collected
            if "mlp" in run_dir.name or "random_forest" in run_dir.name:
                continue

            model = manifest.get("model_name", run_dir.name.split("_")[0])
            parts = run_dir.name.split("_seed")
            seed = int(manifest.get("seed", parts[1] if len(parts) > 1 else 42))
            top_part = parts[0]
            topology = (
                top_part.replace(f"{model}_", "") if top_part.startswith(f"{model}_") else "none"
            )

            rows.append(
                {
                    "run_id": manifest.get("run_id", run_dir.name),
                    "run_type": "sweep",
                    "model": model,
                    "topology": topology,
                    "seed": seed,
                    "val_macro_f1": float(metrics.get("val", {}).get("macro_f1", 0.0)),
                    "test_macro_f1": float(metrics.get("test", {}).get("macro_f1", 0.0)),
                    "test_balanced_acc": float(
                        metrics.get("test", {}).get("balanced_accuracy", 0.0)
                    ),
                    "best_epoch": int(metrics.get("best_epoch", 1)),
                    "duration_seconds": float(
                        metrics.get(
                            "duration_seconds", manifest.get("training_time_seconds", 0.0) or 0.0
                        )
                    ),
                    "manifest_hash": str(
                        manifest.get("model_config_hash", manifest.get("run_hash", ""))
                    ),
                }
            )

    # Sort rows by model, topology, seed
    rows.sort(
        key=lambda r: (str(r["run_type"]), str(r["model"]), str(r["topology"]), int(r["seed"]))
    )

    fieldnames = [
        "run_id",
        "run_type",
        "model",
        "topology",
        "seed",
        "val_macro_f1",
        "test_macro_f1",
        "test_balanced_acc",
        "best_epoch",
        "duration_seconds",
        "manifest_hash",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d run records to %s", len(rows), output_csv)
    print(f"Exported {len(rows)} run records to {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", type=str, default="mouse_held_out_canonical")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    export_runs_to_csv(args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()
