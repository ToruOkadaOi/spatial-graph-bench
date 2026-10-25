"""Regression checks for the corrected metric, inference, integrity and cache contracts."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from spatial_graph_bench.analysis.integrity import verify_paths
from spatial_graph_bench.analysis.reporting import decision_label
from spatial_graph_bench.analysis.statistics import correct_grid, holm_adjust, paired_tests
from spatial_graph_bench.config.model import BenchmarkRunConfig
from spatial_graph_bench.evaluation.metrics import compute_partition_metrics
from spatial_graph_bench.tracking.cache import require_compatible_cache
from spatial_graph_bench.tracking.graph_lift import compute_matched_graph_lift
from spatial_graph_bench.tracking.schema import RunManifest
from spatial_graph_bench.utils.hashing import hash_bytes, hash_dict, hash_file


def test_holm_does_not_restart_rejection_after_first_failure():
    # Independent rank thresholds would incorrectly reject the final 0.049.
    assert holm_adjust([0.04, 0.045, 0.049]) == pytest.approx([0.12, 0.12, 0.12])
    assert holm_adjust([0.03, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])


def test_all_hypotheses_are_corrected_and_inconclusive_is_distinct():
    row = paired_tests([-0.1, 0.1, -0.2, 0.2], 0.005)
    correct_grid([row])
    assert row["family_size"] == 3
    assert row["decision"] == "INCONCLUSIVE"
    assert decision_label(row["decision"]) == "Inconclusive"
    assert decision_label("EQUIVALENT") == "Equivalent"
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_intersection_filters_unseen_true_cells_but_penalizes_unseen_predictions():
    scores = compute_partition_metrics(
        np.array([0, 1, 2]),
        np.array([0, 2, 2]),
        "test",
        {"A": 0, "B": 1, "unseen": 2, "train_only": 3},
        ["A", "B"],
        ["unseen"],
    )
    assert scores.num_samples == 2
    assert scores.macro_f1 == 0.5
    assert scores.label_coverage == pytest.approx(2 / 3)


def test_integrity_does_not_reseal_changed_input(tmp_path):
    file = tmp_path / "labels.npy"
    np.save(file, np.array([0, 1]))
    index = {"files": {"labels.npy": hash_file(file)}}
    verify_paths(tmp_path, index, [file])
    np.save(file, np.array([1, 0]))
    with pytest.raises(ValueError, match="changed artifact"):
        verify_paths(tmp_path, index, [file])


def test_cache_rejects_changed_training_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr("spatial_graph_bench.tracking.cache.source_fingerprint", lambda: "source")
    config = BenchmarkRunConfig(dataset_name="d", split_id="s")
    manifest = SimpleNamespace(
        model_config_hash="old",
        feature_manifest_hash="features",
        graph_artifact_hash="graph",
        source_fingerprint="source",
    )
    features = SimpleNamespace(manifest=SimpleNamespace(compute_manifest_hash=lambda: "features"))
    graph = SimpleNamespace(manifest=SimpleNamespace(compute_manifest_hash=lambda: "graph"))
    with pytest.raises(ValueError, match="model_config_hash"):
        require_compatible_cache(manifest, config, features, graph, tmp_path)


def test_matched_lift_cannot_bypass_different_features():
    manifest = RunManifest(
        run_id="r",
        model_name="mlp",
        model_config_hash="c",
        dataset_name="d",
        split_id="s",
        split_hash="same",
        feature_manifest_hash="a",
        preprocessing_config_hash="same",
        label_mapping_hash="labels",
        seed=42,
    )
    other = manifest.model_copy(update={"feature_manifest_hash": "b"})
    assert manifest.compute_manifest_hash() == hash_dict(
        manifest.model_dump(mode="json", exclude={"source_fingerprint"})
    )
    with pytest.raises(ValueError, match="Feature manifest hash mismatch"):
        compute_matched_graph_lift(other, manifest, 0.8, 0.7)


def test_fresh_recovery_uses_exact_supplement_bytes_and_rejects_changed_originals(tmp_path):
    from scripts.prepare_reanalysis_artifacts import prepare

    name = "artifacts/results/d/s/mlp_none_seed42/test_preds.npy"
    payload = b"frozen-prediction-file"
    files = {name: hash_bytes(payload)}
    (tmp_path / "audits").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "audits/artifact_integrity_v2.json").write_text(
        json.dumps({"files": files, "content_hash": hash_dict(files)})
    )
    (tmp_path / "configs/release_assets.json").write_text("[]")
    with zipfile.ZipFile(tmp_path / "audits/reanalysis_recovery_v2.zip", "w") as archive:
        archive.writestr(name, payload)
    prepare(tmp_path, offline=True)
    assert (tmp_path / name).read_bytes() == payload
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="historical artifact changed"):
        prepare(tmp_path, offline=True)
