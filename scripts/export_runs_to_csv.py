"""Export corrected test metrics, including baselines stored in results directories."""

import argparse
import json
from pathlib import Path

import pandas as pd

from spatial_graph_bench.analysis.reanalysis import analysis_dir
from spatial_graph_bench.utils.hashing import hash_file
from spatial_graph_bench.utils.paths import get_project_root


def export_runs_to_csv(dataset_name: str, split_id: str, output_csv: Path | None = None) -> Path:
    root = get_project_root()
    corrected = json.loads(
        (analysis_dir(root, dataset_name, split_id) / "test_metrics.json").read_text()
    )
    rows = []
    for run_id, test in sorted(corrected.items()):
        run = root / "artifacts/results" / dataset_name / split_id / run_id
        manifest = json.loads((run / "run_manifest.json").read_text())
        old = json.loads((run / "metrics_summary.json").read_text())
        model = manifest["model_name"]
        rows.append(
            {
                "run_id": run_id,
                "run_type": "baseline" if model in ("mlp", "random_forest") else "sweep",
                "model": model,
                "topology": run_id.removeprefix(model + "_").rsplit("_seed", 1)[0],
                "seed": manifest["seed"],
                "val_macro_f1": old["val"]["macro_f1"],
                "val_metric_policy": "historical_unmodified",
                "test_macro_f1": test["macro_f1"],
                "test_balanced_acc": test["balanced_accuracy"],
                "best_epoch": manifest["best_epoch"],
                "duration_seconds": manifest["training_time_seconds"],
                "manifest_hash": hash_file(run / "run_manifest.json"),
                "label_coverage": test["label_coverage"],
                "num_samples": test["num_samples"],
            }
        )
    out = output_csv or root / "results/reports" / f"{dataset_name}_{split_id}_all_runs_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", default="mouse_held_out_canonical")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_runs_to_csv(args.dataset, args.split, args.output))
