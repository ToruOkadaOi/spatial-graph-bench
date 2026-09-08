"""Model training orchestrator for MLP, Random Forest, and GNNs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from spatial_graph_bench.config.model import BenchmarkRunConfig, ModelType
from spatial_graph_bench.evaluation.metrics import compute_partition_metrics
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.models.gnn import BenchmarkGNN
from spatial_graph_bench.models.mlp import MLP
from spatial_graph_bench.models.random_forest import RandomForestBaseline
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.tracking.schema import RunManifest, RunStatus
from spatial_graph_bench.utils.hashing import hash_dict
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.seed import set_seed
from spatial_graph_bench.utils.versioning import get_code_version, get_torch_geometric_version

logger = get_logger("models.trainer")


def run_benchmark_training(
    config: BenchmarkRunConfig,
    feature_bundle: PreprocessedBundle,
    graph_bundle: GraphBundle | None = None,
    out_dir: Path | str | None = None,
) -> tuple[RunManifest, dict[str, Any]]:
    """Execute training for a single configured model run adhering to GPU run contract."""
    start_time = time.time()
    set_seed(config.training.seed)

    device = torch.device(
        config.training.device
        if torch.cuda.is_available() and config.training.device.startswith("cuda")
        else "cpu"
    )

    num_classes = len(feature_bundle.label_to_id)
    in_features = feature_bundle.X_pca_train.shape[1]

    # Evaluated and excluded labels per §3.4
    eval_labels = list(feature_bundle.label_to_id.keys())
    excl_labels: list[str] = []

    # 1. Random Forest Baseline
    if config.model_type == ModelType.RANDOM_FOREST:
        rf = RandomForestBaseline(config.rf_config, random_state=config.training.seed)
        rf.fit(feature_bundle.X_pca_train, feature_bundle.train_labels)

        train_preds = rf.predict(feature_bundle.X_pca_train)
        val_preds = rf.predict(feature_bundle.X_pca_val)
        test_preds = rf.predict(feature_bundle.X_pca_test)
        test_probs = rf.predict_proba(feature_bundle.X_pca_test)

        train_summary = compute_partition_metrics(
            feature_bundle.train_labels,
            train_preds,
            "train",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        val_summary = compute_partition_metrics(
            feature_bundle.val_labels,
            val_preds,
            "val",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        test_summary = compute_partition_metrics(
            feature_bundle.test_labels,
            test_preds,
            "test",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
            section_ids=feature_bundle.section_ids_test,
        )

        history: list[dict[str, float]] = []
        best_val_f1 = val_summary.macro_f1
        best_epoch = 1

    # 2. Tuned MLP Baseline
    elif config.model_type == ModelType.MLP:
        model = MLP(in_features, num_classes, config.mlp_config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        X_tr = torch.tensor(feature_bundle.X_pca_train, dtype=torch.float32, device=device)
        y_tr = torch.tensor(feature_bundle.train_labels, dtype=torch.long, device=device)
        X_va = torch.tensor(feature_bundle.X_pca_val, dtype=torch.float32, device=device)
        X_te = torch.tensor(feature_bundle.X_pca_test, dtype=torch.float32, device=device)

        history = []
        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, config.training.max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tr)
            loss = criterion(logits, y_tr)
            loss.backward()
            optimizer.step()

            # Eval on val
            model.eval()
            with torch.no_grad():
                val_logits = model(X_va)
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
                val_summary = compute_partition_metrics(
                    feature_bundle.val_labels,
                    val_pred,
                    "val",
                    feature_bundle.label_to_id,
                    eval_labels,
                    excl_labels,
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "val_macro_f1": val_summary.macro_f1,
                }
            )

            if val_summary.macro_f1 > best_val_f1 + config.training.min_delta:
                best_val_f1 = val_summary.macro_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            tr_logits = model(X_tr)
            train_preds = tr_logits.argmax(dim=1).cpu().numpy()
            va_logits = model(X_va)
            val_preds = va_logits.argmax(dim=1).cpu().numpy()
            te_logits = model(X_te)
            test_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()
            test_preds = te_logits.argmax(dim=1).cpu().numpy()

        train_summary = compute_partition_metrics(
            feature_bundle.train_labels,
            train_preds,
            "train",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        val_summary = compute_partition_metrics(
            feature_bundle.val_labels,
            val_preds,
            "val",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        test_summary = compute_partition_metrics(
            feature_bundle.test_labels,
            test_preds,
            "test",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
            section_ids=feature_bundle.section_ids_test,
        )

    # 3. GNN Model Suite (GCN, GraphSAGE, GAT, GIN)
    else:
        if graph_bundle is None:
            raise ValueError(f"GraphBundle required for GNN model: {config.model_type}")

        model = BenchmarkGNN(
            model_type=config.model_type,
            in_features=in_features,
            num_classes=num_classes,
            config=config.gnn_config,
        ).to(device)

        X_all = np.vstack(
            [feature_bundle.X_pca_train, feature_bundle.X_pca_val, feature_bundle.X_pca_test]
        )
        pyg_data = graph_bundle.to_pyg_data(X_all, y_train_only=feature_bundle.train_labels).to(
            device
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        tr_mask = pyg_data.train_mask
        val_mask = pyg_data.val_mask
        te_mask = pyg_data.test_mask

        # Invariant: Training labels never seen by validation or test nodes
        y_train_tensor = torch.tensor(feature_bundle.train_labels, dtype=torch.long, device=device)

        history = []
        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, config.training.max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(pyg_data.x, pyg_data.edge_index, getattr(pyg_data, "edge_weight", None))
            loss = criterion(out[tr_mask], y_train_tensor)
            loss.backward()
            optimizer.step()

            # Eval on val
            model.eval()
            with torch.no_grad():
                val_out = out[val_mask]
                val_pred = val_out.argmax(dim=1).cpu().numpy()
                val_summary = compute_partition_metrics(
                    feature_bundle.val_labels,
                    val_pred,
                    "val",
                    feature_bundle.label_to_id,
                    eval_labels,
                    excl_labels,
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "val_macro_f1": val_summary.macro_f1,
                }
            )

            if val_summary.macro_f1 > best_val_f1 + config.training.min_delta:
                best_val_f1 = val_summary.macro_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            final_out = model(
                pyg_data.x, pyg_data.edge_index, getattr(pyg_data, "edge_weight", None)
            )
            train_preds = final_out[tr_mask].argmax(dim=1).cpu().numpy()
            val_preds = final_out[val_mask].argmax(dim=1).cpu().numpy()
            te_out = final_out[te_mask]
            test_probs = torch.softmax(te_out, dim=-1).cpu().numpy()
            test_preds = te_out.argmax(dim=1).cpu().numpy()

        train_summary = compute_partition_metrics(
            feature_bundle.train_labels,
            train_preds,
            "train",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        val_summary = compute_partition_metrics(
            feature_bundle.val_labels,
            val_preds,
            "val",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
        )
        test_summary = compute_partition_metrics(
            feature_bundle.test_labels,
            test_preds,
            "test",
            feature_bundle.label_to_id,
            eval_labels,
            excl_labels,
            section_ids=feature_bundle.section_ids_test,
        )

    elapsed = time.time() - start_time
    run_id = f"{config.model_type.value}_{config.graph_name}_seed{config.training.seed}"

    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCESS,
        model_name=config.model_type.value,
        model_config_hash=config.compute_config_hash(),
        dataset_name=config.dataset_name,
        dataset_version=feature_bundle.manifest.created_at_utc[:10],
        split_id=config.split_id,
        split_hash=feature_bundle.manifest.split_config_hash,
        feature_manifest_hash=feature_bundle.manifest.compute_manifest_hash(),
        preprocessing_config_hash=hash_dict(
            {
                "preprocessing_version": feature_bundle.manifest.preprocessing_version,
                "n_pca_components": feature_bundle.manifest.n_pca_components,
                "n_hvg": feature_bundle.manifest.n_hvg,
                "scale_features": feature_bundle.manifest.scale_features,
            }
        ),
        graph_artifact_hash=graph_bundle.manifest.compute_manifest_hash() if graph_bundle else None,
        label_mapping_hash=feature_bundle.manifest.label_mapping_hash,
        protocol_variant=graph_bundle.manifest.protocol_variant
        if graph_bundle
        else "spatially_ignorant",
        seed=config.training.seed,
        device=str(device),
        code_version=get_code_version(),
        torch_geometric_version=get_torch_geometric_version(),
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        training_time_seconds=elapsed,
    )

    summaries = {
        "train": train_summary.model_dump(),
        "val": val_summary.model_dump(),
        "test": test_summary.model_dump(),
    }

    # Persist artifacts if out_dir given
    if out_dir is not None:
        p = Path(out_dir) / run_id
        p.mkdir(parents=True, exist_ok=True)

        (p / "run_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (p / "metrics_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        np.save(p / "test_preds.npy", test_preds)
        np.save(p / "test_probs.npy", test_probs)

        # Write training_history.csv
        import pandas as pd

        pd.DataFrame(history).to_csv(p / "training_history.csv", index=False)

    return manifest, summaries
