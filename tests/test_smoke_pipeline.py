"""End-to-end CI smoke test running the CPU pipeline on a tiny fixture dataset.

Pipeline sequence:
census check -> split generation & validation -> preprocessing & feature audit ->
spatial graph construction & validation -> 2-epoch model training -> run_manifest verification -> matched lift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scripts.validate_construction import validate_graph_construction
from scripts.validate_split import validate_split_file

from spatial_graph_bench.analysis.delivery import AuditVerdict, audit_run_dir
from spatial_graph_bench.config.graph import SpatialkNNConfig
from spatial_graph_bench.config.model import (
    BenchmarkRunConfig,
    GNNConfig,
    MLPConfig,
    ModelType,
    TrainingConfig,
)
from spatial_graph_bench.config.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipelineVersion,
)
from spatial_graph_bench.config.split import SplitConfig, SplitHierarchy
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.models.trainer import run_benchmark_training
from spatial_graph_bench.preprocessing.audit import verify_spatial_ignorance
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.generator import create_split_definition
from spatial_graph_bench.tracking.graph_lift import compute_matched_graph_lift


def test_end_to_end_smoke_pipeline():
    with tempfile.TemporaryDirectory(prefix="smoke_bench_") as tmp:
        root = Path(tmp)
        dataset_name = "fixture_spatial_ds"
        split_id = "canonical_seed42"

        # 1. Synthesize tiny ST fixture dataset
        n_cells = 45
        n_genes = 25
        rng = np.random.default_rng(42)

        X_counts = sparse.csr_matrix(rng.poisson(lam=4.0, size=(n_cells, n_genes)))
        spatial_coords = rng.uniform(0, 50, size=(n_cells, 2)).astype(np.float32)

        obs = pd.DataFrame(
            {
                "mouse_id": ["mouse1"] * 20 + ["mouse2"] * 12 + ["mouse3"] * 13,
                "section_id": ["sec1"] * 20 + ["sec2"] * 12 + ["sec3"] * 13,
                "cell_type": (["Neuron", "Astrocyte", "Microglia"] * 15)[:n_cells],
            },
            index=[f"spot_{i}" for i in range(n_cells)],
        )

        # 2. Split Generation
        split_cfg = SplitConfig(
            dataset_name=dataset_name,
            split_id=split_id,
            hierarchy=SplitHierarchy.DONOR_HELD_OUT,
            seed=42,
            train_groups=["mouse1"],
            val_groups=["mouse2"],
            test_groups=["mouse3"],
            group_column="mouse_id",
        )
        split = create_split_definition(obs, split_cfg, label_col="cell_type")

        split_file = root / "splits" / dataset_name / f"{split_id}.json"
        split.save_json(split_file)

        # Assert split validation script passes
        assert validate_split_file(split_file) is True

        # 3. Preprocessing & Feature Manifest
        prep_cfg = PreprocessingConfig(
            version=PreprocessingPipelineVersion.STRICT_A,
            n_pca_components=6,
            n_hvg=20,
        )

        # Feature spatial ignorance audit
        assert verify_spatial_ignorance(X_counts, obs, spatial_coords, split, prep_cfg) is True

        prep_bundle = run_feature_pipeline(
            X_counts=X_counts,
            obs=obs,
            spatial_coords=spatial_coords,
            split=split,
            config=prep_cfg,
            section_col="section_id",
        )

        prep_dir = root / "artifacts" / "preprocessed" / dataset_name / split_id
        prep_bundle.save(prep_dir)

        # 4. Graph Construction (Section-Own Spatial k-NN)
        graph_builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4))
        graph_bundle = graph_builder.build(prep_bundle, graph_name="spatial_knn_k4")

        graph_dir = root / "artifacts" / "graphs" / dataset_name / split_id / "spatial_knn_k4"
        graph_bundle.save(graph_dir)

        # Assert graph validation script passes
        assert validate_graph_construction(graph_dir) is True

        # 5. Model Training (2 epochs each)
        results_dir = root / "artifacts" / "results" / dataset_name / split_id

        # 5a. MLP Baseline Run
        mlp_run_cfg = BenchmarkRunConfig(
            model_type=ModelType.MLP,
            dataset_name=dataset_name,
            split_id=split_id,
            graph_name="none",
            mlp_config=MLPConfig(hidden_dims=[32, 16], dropout=0.1),
            training=TrainingConfig(max_epochs=2, seed=42, learning_rate=0.01),
        )
        mlp_manifest, mlp_summaries = run_benchmark_training(
            mlp_run_cfg,
            feature_bundle=prep_bundle,
            out_dir=results_dir,
        )

        # 5b. GCN Model Run
        gcn_run_cfg = BenchmarkRunConfig(
            model_type=ModelType.GCN,
            dataset_name=dataset_name,
            split_id=split_id,
            graph_name="spatial_knn_k4",
            gnn_config=GNNConfig(hidden_dim=32, num_layers=2, dropout=0.1),
            training=TrainingConfig(max_epochs=2, seed=42, learning_rate=0.01),
        )
        gcn_manifest, gcn_summaries = run_benchmark_training(
            gcn_run_cfg,
            feature_bundle=prep_bundle,
            graph_bundle=graph_bundle,
            out_dir=results_dir,
        )

        # 6. Delivery Audit Verification (Layer 1-4)
        mlp_report = audit_run_dir(results_dir / mlp_manifest.run_id, prep_bundle=prep_bundle)
        gcn_report = audit_run_dir(results_dir / gcn_manifest.run_id, prep_bundle=prep_bundle)

        assert mlp_report.verdict in (AuditVerdict.PASS, AuditVerdict.WARN)
        assert gcn_report.verdict in (AuditVerdict.PASS, AuditVerdict.WARN)

        # 7. Matched Graph Lift Computation
        lift_record = compute_matched_graph_lift(
            gnn_manifest=gcn_manifest,
            mlp_manifest=mlp_manifest,
            gnn_macro_f1=gcn_summaries["test"]["macro_f1"],
            mlp_macro_f1=mlp_summaries["test"]["macro_f1"],
            gnn_balanced_acc=gcn_summaries["test"]["balanced_accuracy"],
            mlp_balanced_acc=mlp_summaries["test"]["balanced_accuracy"],
        )

        assert lift_record.is_valid_match is True
        assert lift_record.parity_classification in ("positive", "parity", "negative")
        assert np.isfinite(lift_record.overall_graph_lift)
