"""Explicit v2 byte anchors for immutable historical inputs and predictions.

The v1 PCA manifests sometimes hashed float64 before saving float32. V2 records
the actual saved bytes, retaining all v1 manifests and their discrepancies.
It certifies current artifact identity, not the correctness of historical training.
"""

from __future__ import annotations

import json
from pathlib import Path

from spatial_graph_bench.utils.hashing import hash_dict, hash_file

INDEX = "audits/artifact_integrity_v2.json"


def read_index(root: Path) -> dict:
    data = json.loads((root / INDEX).read_text())
    if data["content_hash"] != hash_dict(data["files"]):
        raise ValueError("V2 integrity index content hash mismatch")
    return data


def verify_paths(root: Path, index: dict, files: list[Path]) -> None:
    for path in files:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        expected = index["files"].get(relative)
        if expected is None or not path.is_file() or hash_file(path) != expected:
            raise ValueError(f"Unanchored or changed artifact: {relative}")


def required_for_analysis(name: str) -> bool:
    return not name.endswith(("/test_probs.npy", "/training_history.csv"))


def verify_index(root: Path, require_all: bool = False) -> dict:
    index = read_index(root)
    paths = [
        root / p
        for p in index["files"]
        if require_all or (root / p).exists() or required_for_analysis(p)
    ]
    verify_paths(root, index, paths)
    index["verified_files_count"] = len(paths)
    return index


def verify_legacy_bundle(directory: Path) -> None:
    """Only accept known v1 discrepancies when every input file is v2-anchored."""
    directory = directory.resolve()
    for root in directory.parents:
        if (root / INDEX).is_file():
            verify_paths(
                root, read_index(root), sorted(p for p in directory.iterdir() if p.is_file())
            )
            return
    raise ValueError(
        f"Legacy feature hashes mismatch and no verified v2 anchor exists: {directory}"
    )
