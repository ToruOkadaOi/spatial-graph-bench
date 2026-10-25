"""Strict-inductive training with fresh validation and resumable neural runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from spatial_graph_bench.config.model import BenchmarkRunConfig, ModelType
from spatial_graph_bench.evaluation.metrics import compute_partition_metrics
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.models.checkpoint import (
    atomic_torch_save,
    environment,
    restore_rng,
    rng_state,
)
from spatial_graph_bench.models.gnn import BenchmarkGNN
from spatial_graph_bench.models.mlp import MLP
from spatial_graph_bench.models.random_forest import RandomForestBaseline
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.tracking.cache import source_fingerprint
from spatial_graph_bench.tracking.schema import RunManifest, RunStatus
from spatial_graph_bench.utils.hashing import hash_array, hash_dict, hash_file
from spatial_graph_bench.utils.seed import set_seed
from spatial_graph_bench.utils.versioning import get_code_version, get_torch_geometric_version

TRAINING_PROTOCOL = "strict_inductive_v3"


def training_identity(config, bundle, graph):
    """Bind resume/cache identity to actual inputs, code and resolved configuration."""
    return {
        "protocol": TRAINING_PROTOCOL,
        "config": config.compute_config_hash(),
        "source": source_fingerprint(),
        "features": bundle.manifest.compute_manifest_hash(),
        "graph": graph.manifest.compute_manifest_hash() if graph else None,
        "arrays": hash_dict(
            {
                f"{kind}_{p}": hash_array(getattr(bundle, attr))
                for p in ("train", "val", "test")
                for kind, attr in (("X", f"X_pca_{p}"), ("y", f"{p}_labels"))
            }
        ),
    }


def run_benchmark_training(
    config: BenchmarkRunConfig,
    feature_bundle: PreprocessedBundle,
    graph_bundle: GraphBundle | None = None,
    out_dir: Path | str | None = None,
    resume: bool = False,
):
    """Train only on training features; select on train/val label intersection.

    The frozen dataset ontology/output dimension is retained across all models.
    Test scores use train/test intersection. Test labels never select epochs.
    Completed directories cannot be overwritten; partial neural runs can resume.
    """
    started = time.monotonic()
    run_id = f"{config.model_type.value}_{config.graph_name}_seed{config.training.seed}"
    output = Path(out_dir) / run_id if out_dir is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    device = torch.device(config.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    set_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    bundle = feature_bundle
    identity = training_identity(config, bundle, graph_bundle)
    env = environment(device)
    checkpoint = Path(out_dir) / ".checkpoints" / f"{run_id}.pt" if output else None
    if checkpoint and checkpoint.exists() and not resume:
        raise FileExistsError(f"Interrupted run exists; pass resume=True: {checkpoint}")

    train_ids = set(bundle.train_labels.tolist())
    eval_labels = {}
    excluded = {}
    for p in ("train", "val", "test"):
        present = set(getattr(bundle, f"{p}_labels").tolist())
        eval_labels[p] = sorted(
            k for k, v in bundle.label_to_id.items() if v in train_ids & present
        )
        excluded[p] = sorted(k for k, v in bundle.label_to_id.items() if v in present - train_ids)
        if not eval_labels[p]:
            raise ValueError(f"No train/{p} label intersection")

    def score(p, predictions):
        return compute_partition_metrics(
            getattr(bundle, f"{p}_labels"),
            predictions,
            p,
            bundle.label_to_id,
            eval_labels[p],
            excluded[p],
            section_ids=getattr(bundle, f"section_ids_{p}"),
        )

    n_train, n_val = len(bundle.train_labels), len(bundle.val_labels)
    num_classes = len(bundle.label_to_id)
    history, best_val_f1, best_epoch, elapsed_before = [], -1.0, 0, 0.0
    if config.model_type == ModelType.RANDOM_FOREST:
        model = RandomForestBaseline(config.rf_config, config.training.seed)
        model.fit(bundle.X_pca_train, bundle.train_labels)
        predictions = {
            p: model.predict(getattr(bundle, f"X_pca_{p}")) for p in ("train", "val", "test")
        }
        test_probs = model.predict_proba(bundle.X_pca_test, num_classes)
        best_val_f1, best_epoch = score("val", predictions["val"]).macro_f1, 1
    else:
        is_mlp = config.model_type == ModelType.MLP
        model = (
            MLP(bundle.X_pca_train.shape[1], num_classes, config.mlp_config)
            if is_mlp
            else BenchmarkGNN(
                config.model_type, bundle.X_pca_train.shape[1], num_classes, config.gnn_config
            )
        ).to(device)
        x_train = torch.as_tensor(bundle.X_pca_train, dtype=torch.float32, device=device)
        # Validation tensors contain training references + validation queries only.
        x_val = torch.as_tensor(
            np.vstack([bundle.X_pca_train, bundle.X_pca_val]), dtype=torch.float32, device=device
        )
        y_train = torch.as_tensor(bundle.train_labels, dtype=torch.long, device=device)
        if not is_mlp:
            if graph_bundle is None:
                raise ValueError("GNN training requires a graph")
            expected_ids = bundle.train_cell_ids + bundle.val_cell_ids + bundle.test_cell_ids
            if graph_bundle.node_cell_ids != expected_ids:
                raise ValueError("Graph and feature node ordering differ")
            if (
                graph_bundle.manifest.feature_manifest_hash
                != bundle.manifest.compute_manifest_hash()
            ):
                raise ValueError("Graph feature parent mismatch")
            edge = graph_bundle.edge_index
            e = edge.numpy()
            partitions = np.concatenate(
                [
                    np.zeros(n_train, dtype=int),
                    np.ones(n_val, dtype=int),
                    np.full(len(bundle.test_labels), 2, dtype=int),
                ]
            )
            sections = np.asarray(
                bundle.section_ids_train + bundle.section_ids_val + bundle.section_ids_test
            )
            protocol = graph_bundle.manifest.protocol_variant
            if protocol == "canonical_section_own":
                if np.any(partitions[e[0]] != partitions[e[1]]) or np.any(
                    sections[e[0]] != sections[e[1]]
                ):
                    raise ValueError("Canonical graph crosses partition/section boundaries")
            elif protocol == "variant_a_bipartite":
                if np.any(e[0] >= n_train):
                    raise ValueError(
                        "Bipartite queries may not send messages to references or other queries"
                    )
            else:
                raise ValueError(f"Unsupported inductive protocol: {protocol}")
            tr_edges = (edge[0] < n_train) & (edge[1] < n_train)
            va_edges = (edge[0] < n_train + n_val) & (edge[1] < n_train + n_val)
            edge_train = edge[:, tr_edges].to(device)
            edge_val = edge[:, va_edges].to(device)
            w = graph_bundle.edge_weight
            weight_train = w[tr_edges].to(device) if w is not None else None
            weight_val = w[va_edges].to(device) if w is not None else None

        def train_forward():
            return model(x_train) if is_mlp else model(x_train, edge_train, weight_train)

        def validation_forward():
            return (
                model(x_val[n_train:]) if is_mlp else model(x_val, edge_val, weight_val)[n_train:]
            )

        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        first_epoch, patience_counter, best_state = 1, 0, None
        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["identity"] != identity or state["environment"] != env:
                raise ValueError("Checkpoint code/config/input/environment mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            first_epoch = state["epoch"] + 1
            history, best_state = state["history"], state["best_state"]
            best_epoch, best_val_f1 = state["best_epoch"], state["best_val_f1"]
            patience_counter, elapsed_before = state["patience_counter"], state["elapsed_seconds"]
            restore_rng(state["rng"])

        for epoch in range(first_epoch, config.training.max_epochs + 1):
            if patience_counter >= config.training.patience:
                break
            model.train()
            optimizer.zero_grad()
            logits = train_forward()
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_predictions = validation_forward().argmax(dim=1).cpu().numpy()
                val_f1 = score("val", val_predictions).macro_f1
            history.append(
                {"epoch": epoch, "train_loss": float(loss.item()), "val_macro_f1": val_f1}
            )
            if val_f1 > best_val_f1 + config.training.min_delta:
                best_val_f1, best_epoch = val_f1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if checkpoint:
                atomic_torch_save(
                    {
                        "identity": identity,
                        "environment": env,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_state": best_state,
                        "best_epoch": best_epoch,
                        "best_val_f1": best_val_f1,
                        "patience_counter": patience_counter,
                        "history": history,
                        "rng": rng_state(),
                        "elapsed_seconds": elapsed_before + time.monotonic() - started,
                    },
                    checkpoint,
                )
        if best_state is None:
            raise RuntimeError("No selected model state")
        model.load_state_dict(best_state)
        model.eval()
        # Test features enter a forward pass only after model selection is complete.
        with torch.no_grad():
            if is_mlp:
                outputs = {
                    p: model(
                        torch.as_tensor(
                            getattr(bundle, f"X_pca_{p}"), dtype=torch.float32, device=device
                        )
                    )
                    for p in ("train", "val", "test")
                }
            else:
                x_all = torch.as_tensor(
                    np.vstack([bundle.X_pca_train, bundle.X_pca_val, bundle.X_pca_test]),
                    dtype=torch.float32,
                    device=device,
                )
                all_out = model(
                    x_all,
                    graph_bundle.edge_index.to(device),
                    graph_bundle.edge_weight.to(device)
                    if graph_bundle.edge_weight is not None
                    else None,
                )
                outputs = {
                    "train": train_forward(),
                    "val": validation_forward(),
                    "test": all_out[n_train + n_val :],
                }
            predictions = {p: value.argmax(dim=1).cpu().numpy() for p, value in outputs.items()}
            test_probs = torch.softmax(outputs["test"], dim=-1).cpu().numpy()

    summaries = {p: score(p, pred).model_dump() for p, pred in predictions.items()}
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCESS,
        model_name=config.model_type.value,
        model_config_hash=config.compute_config_hash(),
        source_fingerprint=identity["source"],
        selected_params=config.model_dump(mode="json"),
        dataset_name=config.dataset_name,
        dataset_version=bundle.manifest.created_at_utc[:10],
        split_id=config.split_id,
        split_hash=bundle.manifest.split_config_hash,
        feature_manifest_hash=bundle.manifest.compute_manifest_hash(),
        preprocessing_config_hash=hash_dict(
            {
                "preprocessing_version": bundle.manifest.preprocessing_version,
                "n_pca_components": bundle.manifest.n_pca_components,
                "n_hvg": bundle.manifest.n_hvg,
                "scale_features": bundle.manifest.scale_features,
            }
        ),
        graph_artifact_hash=graph_bundle.manifest.compute_manifest_hash() if graph_bundle else None,
        label_mapping_hash=bundle.manifest.label_mapping_hash,
        protocol_variant=graph_bundle.manifest.protocol_variant
        if graph_bundle
        else "spatially_ignorant",
        seed=config.training.seed,
        device=str(device),
        code_version=get_code_version(),
        torch_geometric_version=get_torch_geometric_version(),
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        training_time_seconds=elapsed_before + time.monotonic() - started,
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{run_id}-"))
        try:
            (stage / "run_manifest.json").write_text(manifest.model_dump_json(indent=2))
            (stage / "resolved_config.json").write_text(config.model_dump_json(indent=2))
            (stage / "environment.json").write_text(json.dumps(env, indent=2))
            (stage / "performance.json").write_text(
                json.dumps(
                    {
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else None,
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else None,
                        "training_seconds": manifest.training_time_seconds,
                    },
                    indent=2,
                )
            )
            (stage / "metrics_summary.json").write_text(json.dumps(summaries, indent=2))
            pd.DataFrame(history, columns=["epoch", "train_loss", "val_macro_f1"]).to_csv(
                stage / "training_history.csv", index=False
            )
            for p, pred in predictions.items():
                np.save(stage / f"{p}_preds.npy", pred)
            np.save(stage / "test_probs.npy", test_probs)
            if config.model_type == ModelType.RANDOM_FOREST:
                joblib.dump(model.clf, stage / "rf_model.joblib")
            else:
                torch.save(best_state, stage / "model_weights.pt")
            files = {p.name: hash_file(p) for p in sorted(stage.iterdir())}
            (stage / "completed.json").write_text(
                json.dumps({"identity": identity, "files": files}, indent=2)
            )
            os.rename(stage, output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return manifest, summaries
