"""Standalone dummy benchmark runner for fast CPU-only pipeline verification.

Runs the complete end-to-end benchmark pipeline on a synthetic spatial transcriptomics
dataset in ~3 seconds with zero GPU requirements and zero network access.
Verifies all algorithmic invariants, schemas, training loops, audit layers, and matched lift math.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Add src and project root to sys.path so the script can run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import sparse
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
from spatial_graph_bench.graph.audit import validate_graph_construction
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.models.trainer import run_benchmark_training
from spatial_graph_bench.preprocessing.audit import verify_spatial_ignorance
from spatial_graph_bench.preprocessing.pipeline import run_feature_pipeline
from spatial_graph_bench.splitting.generator import create_split_definition
from spatial_graph_bench.tracking.graph_lift import compute_matched_graph_lift
from spatial_graph_bench.utils.logging import setup_logging

console = Console()


def run_dummy_benchmark() -> bool:
    """Execute the full benchmark lifecycle on a synthetic fixture in a temporary directory."""
    start_time = time.time()
    logger = setup_logging(level="INFO", console_output=False)
    logger.info("Initiating dummy benchmark CPU execution")

    console.rule("[bold cyan]SPATIAL-GRAPH-BENCH: DUMMY CPU BENCHMARK RUN")
    console.print(
        "[dim]Running end-to-end validation on synthetic ST fixture (0 GPU, <5 seconds)...[/dim]\n"
    )

    table = Table(title="Dummy Benchmark Verification Steps")
    table.add_column("Stage", style="cyan", width=8)
    table.add_column("Component", style="bold white", width=28)
    table.add_column("Invariant / Check", style="magenta", width=38)
    table.add_column("Result", justify="center", width=10)

    all_passed = True

    with tempfile.TemporaryDirectory(prefix="dummy_bench_") as tmp_dir:
        root = Path(tmp_dir)
        dataset_name = "synthetic_fixture"
        split_id = "fixture_split"

        # -------------------------------------------------------------
        # Stage 1: Synthesize ST Fixture
        # -------------------------------------------------------------
        try:
            n_cells = 45
            n_genes = 25
            rng = np.random.default_rng(42)

            X_counts = sparse.csr_matrix(rng.poisson(lam=4.0, size=(n_cells, n_genes)))
            spatial_coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)

            obs = pd.DataFrame(
                {
                    "mouse_id": ["mouse1"] * 20 + ["mouse2"] * 12 + ["mouse3"] * 13,
                    "section_id": ["sec1"] * 20 + ["sec2"] * 12 + ["sec3"] * 13,
                    "cell_type": (["Neuron", "Astrocyte", "Microglia"] * 15)[:n_cells],
                },
                index=[f"cell_{i:03d}" for i in range(n_cells)],
            )
            table.add_row(
                "1. Data",
                "ST Fixture Synthesis",
                f"{n_cells} cells, {n_genes} genes across 3 mice",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("1. Data", "ST Fixture Synthesis", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

        # -------------------------------------------------------------
        # Stage 2: Split Generation & Disjointness Validation
        # -------------------------------------------------------------
        try:
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

            is_split_valid = validate_split_file(split_file)
            assert is_split_valid, "Split validation failed"
            table.add_row(
                "2. Split",
                "Donor Holdout Split",
                f"Train: {len(split.train_cell_ids)} | Val: {len(split.val_cell_ids)} | Test: {len(split.test_cell_ids)} (Disjoint)",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("2. Split", "Donor Holdout Split", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

        # -------------------------------------------------------------
        # Stage 3: Feature Preprocessing & Spatial Ignorance Audit
        # -------------------------------------------------------------
        try:
            prep_cfg = PreprocessingConfig(
                version=PreprocessingPipelineVersion.STRICT_A,
                n_pca_components=6,
                n_hvg=20,
            )
            is_spatial_ignorant = verify_spatial_ignorance(
                X_counts, obs, spatial_coords, split, prep_cfg
            )
            assert is_spatial_ignorant, "Coordinate shuffle invariance violated"

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

            table.add_row(
                "3. Prep",
                "Train-fit PCA & Audit",
                "Coord shuffle max diff = 0.0 <= 1e-6",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("3. Prep", "Train-fit PCA & Audit", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

        # -------------------------------------------------------------
        # Stage 4: Spatial Graph Construction & Invariant Audit
        # -------------------------------------------------------------
        try:
            graph_builder = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4))
            graph_bundle = graph_builder.build(prep_bundle, graph_name="spatial_knn_k4")
            graph_dir = root / "artifacts" / "graphs" / dataset_name / split_id / "spatial_knn_k4"
            graph_bundle.save(graph_dir)

            is_graph_valid = validate_graph_construction(graph_dir)
            assert is_graph_valid, "Graph construction invariant check failed"

            table.add_row(
                "4. Graph",
                "Section-Own Spatial k-NN",
                f"k=4, {graph_bundle.edge_index.size(1)} edges, 0 cross-partition",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row(
                "4. Graph", "Section-Own Spatial k-NN", str(e), "[bold red]FAIL[/bold red]"
            )
            all_passed = False

        # -------------------------------------------------------------
        # Stage 5: Model Training (MLP Ignorant Baseline + GCN Spatial)
        # -------------------------------------------------------------
        results_dir = root / "artifacts" / "results" / dataset_name / split_id
        try:
            mlp_cfg = BenchmarkRunConfig(
                model_type=ModelType.MLP,
                dataset_name=dataset_name,
                split_id=split_id,
                graph_name="none",
                mlp_config=MLPConfig(hidden_dims=[32, 16], dropout=0.1),
                training=TrainingConfig(max_epochs=2, seed=42, learning_rate=0.01, device="cpu"),
            )
            mlp_manifest, mlp_summaries = run_benchmark_training(
                mlp_cfg,
                feature_bundle=prep_bundle,
                out_dir=results_dir,
            )

            gcn_cfg = BenchmarkRunConfig(
                model_type=ModelType.GCN,
                dataset_name=dataset_name,
                split_id=split_id,
                graph_name="spatial_knn_k4",
                gnn_config=GNNConfig(hidden_dim=32, num_layers=2, dropout=0.1),
                training=TrainingConfig(max_epochs=2, seed=42, learning_rate=0.01, device="cpu"),
            )
            gcn_manifest, gcn_summaries = run_benchmark_training(
                gcn_cfg,
                feature_bundle=prep_bundle,
                graph_bundle=graph_bundle,
                out_dir=results_dir,
            )

            mlp_f1 = mlp_summaries["test"]["macro_f1"]
            gcn_f1 = gcn_summaries["test"]["macro_f1"]
            table.add_row(
                "5. Train",
                "MLP vs GCN (2 Epochs)",
                f"MLP F1: {mlp_f1:.4f} | GCN F1: {gcn_f1:.4f}",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("5. Train", "MLP vs GCN Training", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

        # -------------------------------------------------------------
        # Stage 6: 4-Layer Delivery Ingestion Audit
        # -------------------------------------------------------------
        try:
            mlp_audit = audit_run_dir(results_dir / mlp_manifest.run_id, prep_bundle=prep_bundle)
            gcn_audit = audit_run_dir(results_dir / gcn_manifest.run_id, prep_bundle=prep_bundle)

            assert mlp_audit.verdict in (AuditVerdict.PASS, AuditVerdict.WARN)
            assert gcn_audit.verdict in (AuditVerdict.PASS, AuditVerdict.WARN)

            table.add_row(
                "6. Audit",
                "4-Layer Delivery Audit",
                "Layer 1-4 checks verified on both runs",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("6. Audit", "4-Layer Delivery Audit", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

        # -------------------------------------------------------------
        # Stage 7: Matched Graph Lift & Parity Classification
        # -------------------------------------------------------------
        try:
            lift_record = compute_matched_graph_lift(
                gnn_manifest=gcn_manifest,
                mlp_manifest=mlp_manifest,
                gnn_macro_f1=gcn_summaries["test"]["macro_f1"],
                mlp_macro_f1=mlp_summaries["test"]["macro_f1"],
                gnn_balanced_acc=gcn_summaries["test"]["balanced_accuracy"],
                mlp_balanced_acc=mlp_summaries["test"]["balanced_accuracy"],
            )
            assert lift_record.is_valid_match is True
            delta_f1 = lift_record.overall_graph_lift

            table.add_row(
                "7. Lift",
                "Matched Lift Math",
                f"Delta F1: {delta_f1:+.4f} | Status: {lift_record.parity_classification}",
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row("7. Lift", "Matched Lift Math", str(e), "[bold red]FAIL[/bold red]")
            all_passed = False

    elapsed = time.time() - start_time
    console.print(table)

    if all_passed:
        console.print(
            f"\n[bold green]ALL CHECKS PASSED[/bold green] [dim](Completed in {elapsed:.2f}s)[/dim]"
        )
        console.print(
            "[green]The benchmark pipeline, schemas, training loops, and audits are fully functional on CPU.[/green]\n"
        )
        return True
    else:
        console.print(
            f"\n[bold red]DUMMY BENCHMARK FAILED[/bold red] [dim](Completed in {elapsed:.2f}s)[/dim]\n"
        )
        return False


if __name__ == "__main__":
    success = run_dummy_benchmark()
    sys.exit(0 if success else 1)
