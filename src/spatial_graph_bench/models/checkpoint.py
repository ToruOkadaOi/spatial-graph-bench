"""Atomic, weights-only-compatible checkpoints for interrupted neural runs."""

import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from spatial_graph_bench.utils.versioning import get_torch_geometric_version


def environment(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "pyg": get_torch_geometric_version(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], torch.tensor(n[1].astype(np.int64)), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(handle)
    try:
        torch.save(data, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
