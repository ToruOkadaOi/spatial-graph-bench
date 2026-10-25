"""Scientific regression tests for strict isolation, selection and resumability."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spatial_graph_bench.config.graph import SpatialkNNConfig
from spatial_graph_bench.config.model import (
    BenchmarkRunConfig,
    GNNConfig,
    ModelType,
    TrainingConfig,
)
from spatial_graph_bench.graph.spatial_knn import SpatialkNNGraphBuilder
from spatial_graph_bench.models import trainer
from spatial_graph_bench.models.gnn import BenchmarkGNN
from test_graph_construction import _make_dummy_bundle


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_config(model=ModelType.GRAPHSAGE):
    return BenchmarkRunConfig(
        model_type=model,
        dataset_name="synth",
        split_id="s1",
        graph_name="spatial_knn_k4",
        gnn_config=GNNConfig(hidden_dim=16, num_layers=2, dropout=0.2),
        training=TrainingConfig(max_epochs=4, patience=4, seed=42, device="cpu"),
    )


@pytest.mark.parametrize(
    "architecture", [ModelType.GCN, ModelType.GAT, ModelType.GIN, ModelType.GRAPHSAGE]
)
def test_test_features_cannot_change_fit_or_selected_epoch(tmp_path, architecture, monkeypatch):
    bundle = _make_dummy_bundle()
    graph = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4)).build(bundle, "spatial_knn_k4")
    cfg = make_config(architecture)
    calls = []
    original = BenchmarkGNN.forward

    def observed(self, x, edge_index, edge_weight=None):
        calls.append((self.training, len(x)))
        return original(self, x, edge_index, edge_weight)

    monkeypatch.setattr(BenchmarkGNN, "forward", observed)
    first, metrics = trainer.run_benchmark_training(cfg, bundle, graph, tmp_path / "a")
    assert calls[:8] == [(mode, size) for _ in range(4) for mode, size in [(True, 20), (False, 30)]]
    assert metrics["val"]["macro_f1"] == first.best_val_macro_f1
    changed = copy.deepcopy(bundle)
    changed.X_pca_test += 100
    changed.test_labels = 1 - changed.test_labels
    second, _ = trainer.run_benchmark_training(cfg, changed, graph, tmp_path / "b")
    assert first.best_epoch == second.best_epoch
    run = first.run_id
    a = torch.load(tmp_path / "a" / run / "model_weights.pt", weights_only=True)
    b = torch.load(tmp_path / "b" / run / "model_weights.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "a" / run / "training_history.csv"),
        pd.read_csv(tmp_path / "b" / run / "training_history.csv"),
    )


def test_interrupted_training_resumes_exactly_on_cpu(tmp_path, monkeypatch):
    bundle = _make_dummy_bundle()
    graph = SpatialkNNGraphBuilder(SpatialkNNConfig(k=4)).build(bundle, "spatial_knn_k4")
    cfg = make_config()
    expected, _ = trainer.run_benchmark_training(cfg, bundle, graph, tmp_path / "reference")
    save = trainer.atomic_torch_save

    def interrupt(state, path):
        save(state, path)
        raise RuntimeError("simulated interrupt")

    monkeypatch.setattr(trainer, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        trainer.run_benchmark_training(cfg, bundle, graph, tmp_path / "resumed")
    monkeypatch.setattr(trainer, "atomic_torch_save", save)
    changed = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"learning_rate": 0.002})}
    )
    with pytest.raises(ValueError, match="Checkpoint code/config/input/environment mismatch"):
        trainer.run_benchmark_training(changed, bundle, graph, tmp_path / "resumed", resume=True)
    actual, _ = trainer.run_benchmark_training(
        cfg, bundle, graph, tmp_path / "resumed", resume=True
    )
    assert expected.best_epoch == actual.best_epoch
    run = expected.run_id
    reference = torch.load(tmp_path / "reference" / run / "model_weights.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / run / "model_weights.pt", weights_only=True)
    assert all(torch.equal(reference[k], resumed[k]) for k in reference)
    np.testing.assert_array_equal(
        np.load(tmp_path / "reference" / run / "test_preds.npy"),
        np.load(tmp_path / "resumed" / run / "test_preds.npy"),
    )
    marker = json.loads((tmp_path / "resumed" / run / "completed.json").read_text())
    assert "model_weights.pt" in marker["files"]
    with pytest.raises(FileExistsError):
        trainer.run_benchmark_training(cfg, bundle, graph, tmp_path / "resumed", resume=True)


def test_cuda_request_fails_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = make_config().model_copy(update={"training": TrainingConfig(device="cuda:0")})
    with pytest.raises(RuntimeError, match="silent CPU fallback"):
        trainer.run_benchmark_training(cfg, _make_dummy_bundle(), out_dir=tmp_path)
