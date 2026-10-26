"""Command line for the corrected local/GPU workflow."""

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

from spatial_graph_bench.evaluation.metrics import compute_partition_metrics
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.retraining.inputs import (
    pack_inputs,
    prepare_inputs,
    validate_structure,
    verify_inputs,
)
from spatial_graph_bench.retraining.runner import run_sweep, summarize, verify_completed
from spatial_graph_bench.utils.hashing import hash_file
from spatial_graph_bench.utils.paths import get_project_root


def pack_results(results, archive, full=False):
    if archive.exists():
        raise FileExistsError(archive)
    summarize(results)
    files = {}
    omitted = []
    for marker in results.glob("*/*/*/completed.json"):
        verify_completed(marker.parent)
        for p in sorted(marker.parent.iterdir()):
            if not p.is_file():
                continue
            name = p.relative_to(results).as_posix()
            if not full and p.name == "test_probs.npy":
                omitted.append(name)
                continue
            files[name] = hash_file(p)
    for name in ("experiment.json", "summary.json", "summary.csv"):
        files[name] = hash_file(results / name)
    if full:
        for path in results.rglob("*.pt"):
            if ".checkpoints" in path.parts:
                files[path.relative_to(results).as_posix()] = hash_file(path)
    export = {
        "protocol": "strict_inductive_v3",
        "files": files,
        "omitted": omitted,
        "includes_incomplete_checkpoints": full,
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / "export_manifest.json"
        manifest.write_text(json.dumps(export, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as tar:
            for name in files:
                tar.add(results / name, arcname="results/" + name)
            tar.add(manifest, arcname="results/export_manifest.json")
    digest = hash_file(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Packaged {len(files)} files into {archive}; sha256={digest}")


def verify_results(root, inputs):
    export = json.loads((root / "export_manifest.json").read_text())
    for name, digest in export["files"].items():
        if hash_file(root / name) != digest:
            raise ValueError(f"Returned results changed: {name}")
    print(f"Verified {len(export['files'])} returned files")
    verify_inputs(inputs)
    plan = json.loads((root / "experiment.json").read_text())
    if plan["input_hash"] != verify_inputs(inputs)["input_hash"]:
        raise ValueError("Results belong to different prepared inputs")
    bundles = {}
    for marker in root.glob("*/*/*/completed.json"):
        directory = marker.parent
        manifest = json.loads((directory / "run_manifest.json").read_text())
        ds, sp = manifest["dataset_name"], manifest["split_id"]
        if (ds, sp) not in bundles:
            bundles[(ds, sp)] = PreprocessedBundle.load(inputs / "preprocessed" / ds / sp)
        bundle = bundles[(ds, sp)]
        reported = json.loads((directory / "metrics_summary.json").read_text())
        train = set(bundle.train_labels.tolist())
        for part in ("train", "val", "test"):
            truth = getattr(bundle, f"{part}_labels")
            present = set(truth.tolist())
            labels = sorted(k for k, v in bundle.label_to_id.items() if v in train & present)
            excluded = sorted(k for k, v in bundle.label_to_id.items() if v in present - train)
            actual = compute_partition_metrics(
                truth,
                np.load(directory / f"{part}_preds.npy"),
                part,
                bundle.label_to_id,
                labels,
                excluded,
                section_ids=getattr(bundle, f"section_ids_{part}"),
            )
            previous = reported[part]
            if (
                not np.isclose(actual.macro_f1, previous["macro_f1"], rtol=0, atol=1e-8)
                or not np.isclose(
                    actual.balanced_accuracy, previous["balanced_accuracy"], rtol=0, atol=1e-8
                )
                or actual.num_samples != previous["num_samples"]
                or actual.evaluated_labels != previous["evaluated_labels"]
                or actual.excluded_labels != previous["excluded_labels"]
                or actual.per_class_f1 != previous["per_class_f1"]
                or actual.per_section_macro_f1 != previous["per_section_macro_f1"]
            ):
                raise ValueError(f"Independent metric verification failed: {directory}/{part}")
        if not np.isclose(
            reported["val"]["macro_f1"], manifest["best_val_macro_f1"], rtol=0, atol=1e-8
        ):
            raise ValueError("Selected checkpoint validation score mismatch")
    print("Independently recomputed all returned train/val/test metrics from frozen labels")
    saved_summary = json.loads((root / "summary.json").read_text())
    recomputed_summary = summarize(root, write=False)
    if saved_summary.keys() != recomputed_summary.keys():
        raise ValueError("Returned summary fields differ from recomputed summary")
    if saved_summary["complete_grid"] != recomputed_summary["complete_grid"]:
        raise ValueError("Returned full-grid status differs from recomputed summary")
    saved_rows = saved_summary["comparisons"]
    computed_rows = recomputed_summary["comparisons"]
    if len(saved_rows) != len(computed_rows):
        raise ValueError("Returned comparison count differs from recomputed summary")
    for saved, computed in zip(saved_rows, computed_rows, strict=True):
        if saved.keys() != computed.keys():
            raise ValueError("Returned comparison fields differ from recomputed summary")
        for key, value in computed.items():
            reported = saved[key]
            if isinstance(value, float):
                if not np.isclose(value, reported, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Returned summary value differs: {key}")
            elif value != reported:
                raise ValueError(f"Returned summary value differs: {key}")
    print("Verified returned inference summary without modifying archived files")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--archive", type=Path)
    v = sub.add_parser("verify-inputs")
    v.add_argument("--inputs", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--results", type=Path, required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--phase", choices=["all", "baselines", "gnn"], default="all")
    r.add_argument("--pilot", action="store_true")
    r.add_argument("--datasets", nargs="+")
    r.add_argument("--models", nargs="+")
    r.add_argument("--graphs", nargs="+")
    r.add_argument("--seeds", type=int, nargs="+")
    r.add_argument("--epochs", type=int, default=200)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--allow-dirty", action="store_true", help="Development fixture runs only")
    r = sub.add_parser("summarize")
    r.add_argument("--results", type=Path, required=True)
    r = sub.add_parser("pack")
    r.add_argument("--results", type=Path, required=True)
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument(
        "--full", action="store_true", help="Include probabilities and interruption checkpoints"
    )
    r = sub.add_parser("verify-results")
    r.add_argument("--results", type=Path, required=True)
    r.add_argument("--inputs", type=Path, required=True)
    sub.add_parser("check-gpu")
    args = p.parse_args()
    if args.command == "prepare":
        prepare_inputs(args.source.resolve(), args.output.resolve())
        if args.archive:
            print(pack_inputs(args.output.resolve(), args.archive.resolve()))
    elif args.command == "verify-inputs":
        info = verify_inputs(args.inputs)
        validate_structure(args.inputs)
        print(f"V3 inputs verified: {info['input_hash']}")
    elif args.command == "check-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        x = torch.randn((512, 512), device="cuda", requires_grad=True)
        (x @ x).square().mean().backward()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "torch": str(torch.__version__),
                    "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        root = get_project_root()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]).strip()
        if dirty and not args.allow_dirty:
            raise RuntimeError(
                "Commit changes before training, or use --allow-dirty for development fixtures"
            )
        run_sweep(
            args.inputs,
            args.results,
            args.device,
            args.phase,
            args.pilot,
            args.datasets,
            args.models,
            args.graphs,
            args.seeds,
            args.epochs,
            args.threads,
        )
    elif args.command == "summarize":
        summarize(args.results)
    elif args.command == "pack":
        pack_results(args.results, args.archive, args.full)
    elif args.command == "verify-results":
        verify_results(args.results, args.inputs)
