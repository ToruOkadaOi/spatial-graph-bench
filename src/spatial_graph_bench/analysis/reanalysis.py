"""Re-evaluate frozen predictions without overwriting historical run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_graph_bench.analysis.boundary import compute_boundary_mask
from spatial_graph_bench.analysis.catalog import (
    ANALYSIS_VERSION,
    DATASETS,
    GRAPHS,
    LIMITATION,
    MODELS,
    SEEDS,
)
from spatial_graph_bench.analysis.statistics import correct_grid, paired_tests
from spatial_graph_bench.evaluation.metrics import compute_partition_metrics
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.hashing import hash_dict, hash_file


def analysis_dir(root: Path, dataset: str, split: str) -> Path:
    return root / "results/reports/reanalysis_v2" / dataset / split


def evaluate_dataset(
    root: Path, dataset: str, split_id: str
) -> tuple[dict, list[dict], list[dict]]:
    prep = root / "artifacts/preprocessed" / dataset / split_id
    runs = root / "artifacts/results" / dataset / split_id
    bundle = PreprocessedBundle.load(prep)
    split = SplitDefinition.load_json(root / "splits" / dataset / f"{split_id}.json")
    split.validate_disjointness()
    for partition in ("train", "val", "test"):
        if getattr(bundle, f"{partition}_cell_ids") != getattr(split, f"{partition}_cell_ids"):
            raise ValueError(f"Cell order differs from frozen split: {dataset}/{partition}")
    train_ids = set(bundle.train_labels.tolist())
    test_ids = set(bundle.test_labels.tolist())
    expected = sorted(k for k, v in bundle.label_to_id.items() if v in train_ids & test_ids)
    excluded = sorted(k for k, v in bundle.label_to_id.items() if v in test_ids - train_ids)
    if expected != split.evaluated_labels or excluded != split.excluded_labels:
        raise ValueError(f"Frozen split label policy differs from arrays: {dataset}")
    if bundle.manifest.split_config_hash != split.compute_artifact_hash():
        raise ValueError(f"Feature/split parent hash mismatch: {dataset}")
    boundary = compute_boundary_mask(bundle.spatial_test, np.array(bundle.section_ids_test))
    run_rows, summaries, source_hashes = [], {}, {}
    graph_manifests = {
        graph: GraphBundle.load(root / "artifacts/graphs" / dataset / split_id / graph).manifest
        for graph in GRAPHS
    }
    run_ids = [f"mlp_none_seed{s}" for s in SEEDS] + ["random_forest_none_seed42"]
    run_ids += [f"{m}_{g}_seed{s}" for m in MODELS for g in GRAPHS for s in SEEDS]
    for run_id in run_ids:
        directory = runs / run_id
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if (manifest["dataset_name"], manifest["split_id"], manifest["run_id"]) != (
            dataset,
            split_id,
            run_id,
        ):
            raise ValueError(f"Run identity mismatch: {directory}")
        if manifest["feature_manifest_hash"] != bundle.manifest.compute_manifest_hash():
            raise ValueError(f"Run feature identity mismatch: {directory}")
        if manifest["split_hash"] != bundle.manifest.split_config_hash or manifest[
            "label_mapping_hash"
        ] != hash_dict(bundle.label_to_id):
            raise ValueError(f"Run split/label identity mismatch: {directory}")
        predictions = np.load(directory / "test_preds.npy")
        if (
            predictions.shape != bundle.test_labels.shape
            or not np.isin(predictions, list(bundle.label_to_id.values())).all()
        ):
            raise ValueError(f"Invalid test predictions: {directory}")
        model = manifest["model_name"]
        graph = run_id.removeprefix(model + "_").rsplit("_seed", 1)[0]
        if (
            model in MODELS
            and manifest["graph_artifact_hash"] != graph_manifests[graph].compute_manifest_hash()
        ):
            raise ValueError(f"Run graph identity mismatch: {directory}")
        score = compute_partition_metrics(
            bundle.test_labels,
            predictions,
            "test",
            bundle.label_to_id,
            expected,
            excluded,
            bundle.section_ids_test,
            boundary,
        )
        summaries[run_id] = score.model_dump(exclude={"confusion_matrix"})
        original = json.loads((directory / "metrics_summary.json").read_text())
        run_rows.append(
            {
                "run_id": run_id,
                "run_type": "sweep" if model in MODELS else "baseline",
                "model": model,
                "topology": graph,
                "seed": manifest["seed"],
                "val_macro_f1": original["val"]["macro_f1"],
                "val_metric_policy": "historical_unmodified",
                "test_macro_f1": score.macro_f1,
                "test_balanced_acc": score.balanced_accuracy,
                "best_epoch": manifest["best_epoch"],
                "duration_seconds": manifest["training_time_seconds"],
                "manifest_hash": hash_file(directory / "run_manifest.json"),
                "label_coverage": score.label_coverage,
                "num_samples": score.num_samples,
            }
        )
        for name in ("run_manifest.json", "metrics_summary.json", "test_preds.npy"):
            file = directory / name
            source_hashes[str(file.relative_to(root))] = hash_file(file)
    mlp = np.array([summaries[f"mlp_none_seed{s}"]["macro_f1"] for s in SEEDS])
    epsilon = max(0.005, float(2 * mlp.std(ddof=1)))
    from spatial_graph_bench.analysis.integrity import read_index
    from spatial_graph_bench.tracking.cache import source_fingerprint

    info = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source_fingerprint": source_fingerprint(),
        "artifact_index_hash": read_index(root)["content_hash"],
        "dataset_name": dataset,
        "split_id": split_id,
        "test_cells": len(bundle.test_labels),
        "evaluated_cells": summaries[run_ids[0]]["num_samples"],
        "evaluated_labels": expected,
        "excluded_labels": excluded,
        "test_label_support": {
            k: int((bundle.test_labels == v).sum()) for k, v in bundle.label_to_id.items()
        },
        "label_coverage": split.coverage_ratio,
        "train_groups": split.train_groups,
        "val_groups": split.val_groups,
        "test_groups": split.test_groups,
        "num_mlp_seeds": len(SEEDS),
        "mlp_mean_test_macro_f1": float(mlp.mean()),
        "mlp_sigma_test_macro_f1": float(mlp.std(ddof=1)),
        "parity_band_halfwidth": float(2 * mlp.std(ddof=1)),
        "tost_equivalence_margin_epsilon": epsilon,
        "rf_test_macro_f1": summaries["random_forest_none_seed42"]["macro_f1"],
        "limitation": LIMITATION,
        "source_hashes": source_hashes,
    }
    rows = []
    for model in MODELS:
        for graph in GRAPHS:
            scores = [summaries[f"{model}_{graph}_seed{s}"] for s in SEEDS]
            baselines = [summaries[f"mlp_none_seed{s}"] for s in SEEDS]
            paired = [g["macro_f1"] - b["macro_f1"] for g, b in zip(scores, baselines, strict=True)]
            row = dict(
                dataset_name=dataset,
                split_id=split_id,
                model=model.upper(),
                graph=graph,
                protocol_variant="variant_a_bipartite"
                if graph.startswith("bipartite")
                else "canonical_section_own",
                epsilon=epsilon,
                paired_lifts=paired,
                seeds=list(SEEDS),
                gnn_overall=float(np.mean([s["macro_f1"] for s in scores])),
                mlp_overall=float(mlp.mean()),
                **paired_tests(paired, epsilon),
            )
            for short, key in (("int", "interior_macro_f1"), ("bnd", "boundary_macro_f1")):
                row[f"gnn_{short}"] = float(np.mean([s[key] for s in scores]))
                row[f"{short}_lift_mean"] = float(
                    np.mean([g[key] - b[key] for g, b in zip(scores, baselines, strict=True)])
                )
            rows.append(row)
    output = analysis_dir(root, dataset, split_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return info, rows, run_rows


def run_reanalysis(root: Path) -> list[dict]:
    """Require the entire frozen grid so the correction family cannot shrink silently."""
    collected = [evaluate_dataset(root, ds, sp) for ds, sp, _ in DATASETS]
    all_rows = [r for _, rows, _ in collected for r in rows]
    correct_grid(all_rows)
    for (ds, sp, _), (info, rows, run_rows) in zip(DATASETS, collected, strict=True):
        output = analysis_dir(root, ds, sp)
        (output / "baseline_summary.json").write_text(json.dumps(info, indent=2) + "\n")
        (output / "post_hoc_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        pd.DataFrame(run_rows).to_csv(
            root / "results/reports" / f"{ds}_{sp}_all_runs_summary.csv", index=False
        )
    return all_rows


def regenerate():
    from spatial_graph_bench.analysis.integrity import verify_index
    from spatial_graph_bench.analysis.reporting import generate_reports
    from spatial_graph_bench.utils.paths import get_project_root

    root = get_project_root()
    index = verify_index(root)
    print(f"Verified {index['verified_files_count']} immutable historical files", flush=True)
    rows = run_reanalysis(root)
    generate_reports(root)
    print(
        f"Regenerated {len(rows)} comparisons; global family has {len(rows) * 3} hypotheses",
        flush=True,
    )
