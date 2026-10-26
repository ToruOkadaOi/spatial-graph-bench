"""Exercise the worker lifecycle on a small CPU fixture, including export/reuse."""

import json
import tarfile

import pytest

from spatial_graph_bench.config.graph import SpatialkNNConfig
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.retraining import runner
from spatial_graph_bench.retraining.cli import pack_results, verify_results
from spatial_graph_bench.utils.hashing import hash_dict, hash_file
from test_graph_construction import _make_dummy_bundle


def test_pilot_skip_export_and_tamper_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DATASETS", [("synth", "s1", "fixture")])
    inputs = tmp_path / "inputs"
    bundle = _make_dummy_bundle()
    bundle.save(inputs / "preprocessed/synth/s1")
    graph = SpatialkNNGraphBuilder(SpatialkNNConfig(k=6)).build(bundle, "spatial_knn_k6")
    graph.save(inputs / "graphs/synth/s1/spatial_knn_k6")
    files = {
        p.relative_to(inputs).as_posix(): hash_file(p) for p in inputs.rglob("*") if p.is_file()
    }
    (inputs / "prepared.json").write_text(
        json.dumps(
            {"protocol": "strict_inductive_v3", "files": files, "input_hash": hash_dict(files)}
        )
    )
    output = tmp_path / "pilot"
    kwargs = {
        "device": "cpu",
        "pilot": True,
        "models": ["graphsage"],
        "graphs": ["spatial_knn_k6"],
        "seeds": [42, 43],
        "epochs": 2,
        "threads": 1,
    }
    runner.run_sweep(inputs, output, **kwargs)
    markers = list(output.glob("*/*/*/completed.json"))
    assert len(markers) == 4
    times = {p: p.stat().st_mtime_ns for p in markers}
    runner.run_sweep(inputs, output, **kwargs)
    assert all(p.stat().st_mtime_ns == times[p] for p in markers)
    archive = tmp_path / "pilot.tar.gz"
    pack_results(output, archive)
    received = tmp_path / "received"
    with tarfile.open(archive) as tar:
        tar.extractall(received, filter="data")
    verify_results(received / "results", inputs)
    summary_path = received / "results/summary.json"
    original_summary_hash = hash_file(summary_path)
    paired_tests = runner.paired_tests

    def platform_rounding(lifts, epsilon):
        values = paired_tests(lifts, epsilon)
        values["ci_90_high"] += 1e-15
        return values

    with monkeypatch.context() as patch:
        patch.setattr(runner, "paired_tests", platform_rounding)
        verify_results(received / "results", inputs)
    assert hash_file(summary_path) == original_summary_hash
    summary = json.loads((received / "results/summary.json").read_text())
    assert summary["complete_grid"] is False
    assert all(r["decision"].startswith("DESCRIPTIVE") for r in summary["comparisons"])
    assert not list((received / "results").glob("*/*/*/test_probs.npy"))
    returned = received / "results"
    metric = next(returned.glob("*/*/*/metrics_summary.json"))
    data = json.loads(metric.read_text())
    data["test"]["macro_f1"] += 0.1
    metric.write_text(json.dumps(data))
    marker = metric.parent / "completed.json"
    data = json.loads(marker.read_text())
    data["files"][metric.name] = hash_file(metric)
    marker.write_text(json.dumps(data))
    export = returned / "export_manifest.json"
    data = json.loads(export.read_text())
    for file in (metric, marker):
        data["files"][file.relative_to(returned).as_posix()] = hash_file(file)
    export.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Independent metric verification failed"):
        verify_results(returned, inputs)
    bad = markers[0].parent / "test_preds.npy"
    bad.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed/missing"):
        runner.run_sweep(inputs, output, **kwargs)
