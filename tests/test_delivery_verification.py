"""Unit tests for GPU results packaging and 4-layer verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from spatial_graph_bench.analysis.delivery import (
    AuditVerdict,
    audit_run_dir,
    build_batch_manifest,
    verify_batch_files,
)
from spatial_graph_bench.tracking.schema import RunManifest


def test_delivery_verification_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_01"
        run_dir.mkdir()

        manifest = RunManifest(
            run_id="run_01",
            model_name="mlp",
            model_config_hash="cfg123",
            dataset_name="test_ds",
            split_id="s1",
            feature_manifest_hash="feat123",
            label_mapping_hash="label123",
            seed=42,
        )
        (run_dir / "run_manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

        y_pred = np.array([0, 1, 0, 1])
        y_probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
        np.save(run_dir / "test_preds.npy", y_pred)
        np.save(run_dir / "test_probs.npy", y_probs)

        summaries = {
            "test": {
                "partition": "test",
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "num_samples": 4,
            }
        }
        (run_dir / "metrics_summary.json").write_text(json.dumps(summaries), encoding="utf-8")

        report = audit_run_dir(run_dir)
        # Without local preprocessed bundle, verdict is warn or pass
        assert report.verdict in (AuditVerdict.PASS, AuditVerdict.WARN)

        # Build batch manifest
        batch_manifest = build_batch_manifest(Path(tmp), "test_ds", "s1", [report])
        assert len(batch_manifest.files) > 0

        # Verify batch files (Layer 1)
        problems, extras = verify_batch_files(Path(tmp), batch_manifest)
        assert len(problems) == 0

        # Simulate corruption (Layer 1 failure)
        (run_dir / "test_preds.npy").write_bytes(b"corrupted_bytes")
        problems_corrupt, _ = verify_batch_files(Path(tmp), batch_manifest)
        assert len(problems_corrupt) > 0
