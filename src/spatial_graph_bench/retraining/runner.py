"""Run isolated pilot/full experiments with verified completed-run reuse."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from spatial_graph_bench.analysis.catalog import DATASETS, GRAPHS, MODELS, SEEDS
from spatial_graph_bench.analysis.statistics import correct_grid, paired_tests
from spatial_graph_bench.config.model import (
    BenchmarkRunConfig,
    GNNConfig,
    MLPConfig,
    ModelType,
    RandomForestConfig,
    TrainingConfig,
)
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.models.trainer import run_benchmark_training, training_identity
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.retraining.inputs import verify_inputs
from spatial_graph_bench.tracking.cache import source_fingerprint
from spatial_graph_bench.utils.hashing import hash_file


def verify_completed(directory: Path, identity=None, omitted=()):
    marker = json.loads((directory / "completed.json").read_text())
    if identity is not None and marker["identity"] != identity:
        raise ValueError(f"Completed run identity differs: {directory}")
    for name, digest in marker["files"].items():
        if name in omitted:
            continue
        if not (directory / name).is_file() or hash_file(directory / name) != digest:
            raise ValueError(f"Completed run file changed/missing: {directory / name}")
    return json.loads((directory / "metrics_summary.json").read_text())


def _run_sweep(
    inputs,
    results,
    device="cuda:0",
    phase="all",
    pilot=False,
    datasets=None,
    models=None,
    graphs=None,
    seeds=None,
    epochs=200,
    threads=4,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Check the driver/runtime before starting a paid sweep."
        )
    if results.resolve().is_relative_to(inputs.resolve()):
        raise ValueError("Results must be outside the frozen input directory")
    if not 1 <= epochs <= 2000 or threads < 1:
        raise ValueError("Invalid epoch/thread budget")
    torch.set_num_threads(threads)
    manifest = verify_inputs(inputs)
    selected = datasets or ([d[0] for d in DATASETS[:2]] if pilot else [d[0] for d in DATASETS])
    unknown = set(selected) - {d[0] for d in DATASETS}
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    models = models or (["gcn", "graphsage"] if pilot else list(MODELS))
    graphs = graphs or (["spatial_knn_k12"] if pilot else list(GRAPHS))
    seeds = seeds or ([42, 43, 44] if pilot else list(SEEDS))
    if set(models) - set(MODELS) or set(graphs) - set(GRAPHS):
        raise ValueError("Unknown architecture or graph")
    if len(set(seeds)) != len(seeds) or len(set(selected)) != len(selected):
        raise ValueError("Duplicate seeds/datasets are not allowed")
    feature_hashes = {
        ds: PreprocessedBundle.load(
            inputs / "preprocessed" / ds / sp
        ).manifest.compute_manifest_hash()
        for ds, sp, _ in DATASETS
        if ds in selected
    }
    plan = {
        "protocol": "strict_inductive_v3",
        "source_fingerprint": source_fingerprint(),
        "feature_hashes": feature_hashes,
        "input_hash": manifest["input_hash"],
        "pilot": pilot,
        "datasets": selected,
        "models": models,
        "graphs": graphs,
        "seeds": seeds,
        "max_epochs": epochs,
        "validation_policy": "train_val_intersection",
        "threads": threads,
    }
    # Permit baseline-only work locally followed by GNN work on the GPU;
    # all immutable scientific settings must still agree.
    results.mkdir(parents=True, exist_ok=True)
    plan_file = results / "experiment.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Experiment settings differ; select a new results directory")
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    for ds, sp, _ in DATASETS:
        if ds not in selected:
            continue
        bundle = PreprocessedBundle.load(inputs / "preprocessed" / ds / sp)
        out = results / ds / sp
        graph_cache = {}
        if phase in ("gnn", "all"):
            graph_cache = {g: GraphBundle.load(inputs / "graphs" / ds / sp / g) for g in graphs}
        jobs = []
        if phase in ("baselines", "all"):
            jobs += [(ModelType.MLP, "none", s) for s in seeds]
            if not pilot:
                jobs.append((ModelType.RANDOM_FOREST, "none", 42))
        if phase in ("gnn", "all"):
            for seed in seeds:
                directory = out / f"mlp_none_seed{seed}"
                # When phase=all these will be created before the first GNN job.
                if phase == "gnn":
                    scores = verify_completed(directory)
                    bman = json.loads((directory / "run_manifest.json").read_text())
                    if (
                        bman["feature_manifest_hash"] != bundle.manifest.compute_manifest_hash()
                        or not scores["test"]["evaluated_labels"]
                        or bman["source_fingerprint"] != plan["source_fingerprint"]
                        or json.loads((directory / "completed.json").read_text())["identity"][
                            "protocol"
                        ]
                        != "strict_inductive_v3"
                    ):
                        raise ValueError("Baseline/input mismatch")
            jobs += [(ModelType(m), g, s) for m in models for g in graphs for s in seeds]
        for kind, graph, seed in jobs:
            config = BenchmarkRunConfig(
                model_type=kind,
                dataset_name=ds,
                split_id=sp,
                graph_name=graph,
                mlp_config=MLPConfig(hidden_dims=[256, 128], dropout=0.2, use_batch_norm=True)
                if kind == ModelType.MLP
                else None,
                rf_config=RandomForestConfig(n_jobs=threads)
                if kind == ModelType.RANDOM_FOREST
                else None,
                gnn_config=GNNConfig(hidden_dim=128, num_layers=2, dropout=0.2)
                if kind in [ModelType(m) for m in MODELS]
                else None,
                training=TrainingConfig(
                    max_epochs=epochs,
                    patience=15,
                    seed=seed,
                    device="cpu" if kind == ModelType.RANDOM_FOREST else device,
                ),
                extra_params={"training_protocol": "strict_inductive_v3"},
            )
            g = graph_cache.get(graph)
            run_id = f"{kind.value}_{graph}_seed{seed}"
            directory = out / run_id
            if directory.exists():
                verify_completed(directory, training_identity(config, bundle, g))
                print(f"SKIP completed {ds}/{run_id}", flush=True)
                continue
            print(f"TRAIN {ds}/{run_id}", flush=True)
            run_benchmark_training(config, bundle, g, out, resume=True)
            print(f"DONE {ds}/{run_id}", flush=True)
    print(
        "Sweep phase complete. Re-running the identical command resumes unfinished work.",
        flush=True,
    )


def summarize(results: Path, *, write: bool = True):
    plan = json.loads((results / "experiment.json").read_text())
    export = (
        json.loads((results / "export_manifest.json").read_text())
        if (results / "export_manifest.json").exists()
        else None
    )
    omitted = {Path(p).parts[-1] for p in export["omitted"]} if export else set()
    records = []
    for ds, sp, _ in DATASETS:
        if ds not in plan["datasets"]:
            continue
        base = {}
        for seed in plan["seeds"]:
            directory = results / ds / sp / f"mlp_none_seed{seed}"
            if not directory.exists():
                continue
            baseline = verify_completed(directory, omitted=omitted)
            marker = json.loads((directory / "completed.json").read_text())["identity"]
            if (
                marker["source"] != plan["source_fingerprint"]
                or marker["features"] != plan["feature_hashes"][ds]
            ):
                raise ValueError("Summary baseline belongs to another experiment")
            base[seed] = baseline["test"]["macro_f1"]
        for m in plan["models"]:
            for g in plan["graphs"]:
                pairs = []
                scores = []
                matched = []
                for seed in plan["seeds"]:
                    d = results / ds / sp / f"{m}_{g}_seed{seed}"
                    if not d.exists() or seed not in base:
                        continue
                    value = verify_completed(d, omitted=omitted)["test"]["macro_f1"]
                    marker = json.loads((d / "completed.json").read_text())["identity"]
                    bmarker = json.loads(
                        (results / ds / sp / f"mlp_none_seed{seed}" / "completed.json").read_text()
                    )["identity"]
                    if any(
                        marker[k] != bmarker[k]
                        for k in ("protocol", "source", "features", "arrays")
                    ):
                        raise ValueError("Summary cannot pair different inputs/training protocols")
                    pairs.append(value - base[seed])
                    scores.append(value)
                    matched.append(base[seed])
                if len(pairs) < 2:
                    continue
                epsilon = max(0.005, float(2 * np.std(list(base.values()), ddof=1)))
                records.append(
                    {
                        "dataset": ds,
                        "split": sp,
                        "model": m,
                        "graph": g,
                        "gnn_macro_f1": float(np.mean(scores)),
                        "mlp_macro_f1": float(np.mean(matched)),
                        "epsilon": epsilon,
                        **paired_tests(pairs, epsilon),
                    }
                )
    complete = (
        not plan["pilot"]
        and set(plan["datasets"]) == {d[0] for d in DATASETS}
        and set(plan["models"]) == set(MODELS)
        and set(plan["graphs"]) == set(GRAPHS)
        and plan["seeds"] == list(SEEDS)
        and len(records) == 112
        and all(r["n"] == 10 for r in records)
    )
    if complete:
        correct_grid(records)
    else:
        for row in records:
            row["decision"] = "DESCRIPTIVE ONLY — incomplete full grid"
    summary = {"complete_grid": complete, "comparisons": records}
    if write:
        (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        pd.DataFrame(records).to_csv(results / "summary.csv", index=False)
    print(f"Summarized {len(records)} comparisons; full-grid inference={complete}")
    return summary


def run_sweep(inputs, results, *args, **kwargs):
    results.mkdir(parents=True, exist_ok=True)
    with FileLock(results / ".sweep.lock", timeout=0):
        return _run_sweep(inputs, results, *args, **kwargs)
