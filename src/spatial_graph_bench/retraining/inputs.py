"""Prepare portable v3 inputs from frozen v1 features without fitting preprocessing."""

import json
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from spatial_graph_bench.analysis.catalog import DATASETS, GRAPHS
from spatial_graph_bench.config.graph import CoordinateShuffleConfig, RewiredControlConfig
from spatial_graph_bench.graph.coordinate_shuffle import build_coordinate_shuffled_graph
from spatial_graph_bench.graph.rewired_control import build_rewired_control_graph
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.tracking.cache import source_fingerprint
from spatial_graph_bench.utils.hashing import hash_array, hash_dict, hash_file


def verify_inputs(root: Path):
    manifest = json.loads((root / "prepared.json").read_text())
    if (
        manifest["protocol"] != "strict_inductive_v3"
        or hash_dict(manifest["files"]) != manifest["input_hash"]
    ):
        raise ValueError("V3 input manifest invalid")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file() or hash_file(path) != digest:
            raise ValueError(f"Input bytes changed or missing: {name}")
    return manifest


def prepare_inputs(source: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(
            f"Input destination exists: {destination}; verify it or use a new directory"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".preparing-"))
    origins = {}
    try:
        for ds, sp, _ in DATASETS:
            print(f"Preparing {ds}/{sp}", flush=True)
            original = source / "artifacts/preprocessed" / ds / sp
            bundle = PreprocessedBundle.load(original)
            old_parent = bundle.manifest.compute_manifest_hash()
            content = {}
            for part in ("train", "val", "test"):
                for kind, attr in (
                    ("X_pca", f"X_pca_{part}"),
                    ("spatial", f"spatial_{part}"),
                    ("labels", f"{part}_labels"),
                ):
                    content[f"{kind}_{part}"] = hash_array(getattr(bundle, attr))
            content["cell_ids"] = hash_dict(json.loads((original / "cell_ids.json").read_text()))
            bundle.manifest = bundle.manifest.model_copy(
                update={
                    **{f"pca_{p}_hash": content[f"X_pca_{p}"] for p in ("train", "val", "test")},
                    "content_hashes": content,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            bundle.save(stage / "preprocessed" / ds / sp)
            origins[ds] = {
                "v1_feature_manifest_hash": old_parent,
                "persisted_content_hashes": content,
            }
            split_dest = stage / "splits" / ds / f"{sp}.json"
            split_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / "splits" / ds / f"{sp}.json", split_dest)
            base_graphs = {}
            for graph in ("spatial_knn_k6", "spatial_knn_k12", "bipartite_ref_k20"):
                g = GraphBundle.load(source / "artifacts/graphs" / ds / sp / graph)
                if g.manifest.feature_manifest_hash != old_parent:
                    raise ValueError(f"Original graph parent mismatch: {ds}/{graph}")
                g.manifest = g.manifest.model_copy(
                    update={"feature_manifest_hash": bundle.manifest.compute_manifest_hash()}
                )
                g.save(stage / "graphs" / ds / sp / graph)
                base_graphs[graph] = g
            for k in (6, 12):
                name = f"rewired_spatial_knn_k{k}"
                g = build_rewired_control_graph(
                    base_graphs[f"spatial_knn_k{k}"], RewiredControlConfig(seed=42), name
                )
                g.save(stage / "graphs" / ds / sp / name)
                name = f"shuffled_spatial_knn_k{k}"
                g = build_coordinate_shuffled_graph(
                    bundle, CoordinateShuffleConfig(seed=42), name, k=k
                )
                g.save(stage / "graphs" / ds / sp / name)
        files = {
            p.relative_to(stage).as_posix(): hash_file(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "protocol": "strict_inductive_v3",
            "source_fingerprint": source_fingerprint(),
            "control_seed": 42,
            "undirected_swap_attempts_per_edge": 10,
            "origins": origins,
            "files": files,
            "input_hash": hash_dict(files),
        }
        (stage / "prepared.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_structure(stage)
        os.rename(stage, destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_structure(root: Path):
    """Validate actual graph invariants and label/split/node alignment."""
    from spatial_graph_bench.splitting.schema import SplitDefinition

    for ds, sp, _ in DATASETS:
        bundle = PreprocessedBundle.load(root / "preprocessed" / ds / sp)
        split = SplitDefinition.load_json(root / "splits" / ds / f"{sp}.json")
        if split.compute_artifact_hash() != bundle.manifest.split_config_hash:
            raise ValueError("Split identity mismatch")
        for p in ("train", "val", "test"):
            if getattr(bundle, f"{p}_cell_ids") != getattr(split, f"{p}_cell_ids"):
                raise ValueError("Split order mismatch")
        ntr, nva = len(bundle.train_labels), len(bundle.val_labels)
        parts = np.zeros(
            len(bundle.train_labels) + len(bundle.val_labels) + len(bundle.test_labels), dtype=int
        )
        parts[ntr : ntr + nva] = 1
        parts[ntr + nva :] = 2
        sections = np.array(
            bundle.section_ids_train + bundle.section_ids_val + bundle.section_ids_test
        )
        ids = bundle.train_cell_ids + bundle.val_cell_ids + bundle.test_cell_ids
        for name in GRAPHS:
            g = GraphBundle.load(root / "graphs" / ds / sp / name)
            e = g.edge_index.numpy()
            if (
                g.node_cell_ids != ids
                or g.node_section_ids != sections.tolist()
                or g.manifest.feature_manifest_hash != bundle.manifest.compute_manifest_hash()
            ):
                raise ValueError("Graph node order or parent mismatch")
            if name.startswith("bipartite"):
                if np.any(e[0] >= ntr):
                    raise ValueError("Query-to-reference edge")
            elif np.any(parts[e[0]] != parts[e[1]]) or np.any(sections[e[0]] != sections[e[1]]):
                raise ValueError("Disallowed graph edge")
            if name.startswith("rewired"):
                edges = set(map(tuple, e.T.tolist()))
                if any((v, u) not in edges for u, v in edges):
                    raise ValueError("Asymmetric rewiring")
                base = GraphBundle.load(root / "graphs" / ds / sp / name.removeprefix("rewired_"))
                if not np.array_equal(
                    np.bincount(e[0], minlength=g.num_nodes),
                    np.bincount(base.edge_index[0].numpy(), minlength=g.num_nodes),
                ):
                    raise ValueError("Rewiring changed degrees")


def pack_inputs(root: Path, archive: Path):
    verify_inputs(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname="inputs")
    digest = hash_file(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    return digest
