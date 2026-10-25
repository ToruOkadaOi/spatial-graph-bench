"""Validate actual feature/graph bytes, frozen split and node ordering."""

import argparse

from spatial_graph_bench.analysis.catalog import GRAPHS
from spatial_graph_bench.analysis.integrity import INDEX, read_index, verify_paths
from spatial_graph_bench.graph.audit import validate_graph_construction
from spatial_graph_bench.graph.schema import GraphBundle
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.paths import ArtifactPaths


def validate_all_artifacts(dataset_name: str, split_id: str) -> bool:
    paths = ArtifactPaths.default()
    try:
        split_file = paths.dataset_split_file(dataset_name, split_id)
        split = SplitDefinition.load_json(split_file)
        prep = paths.dataset_preprocessed_dir(dataset_name, split_id)
        bundle = PreprocessedBundle.load(prep)
        if bundle.manifest.split_config_hash != split.compute_artifact_hash():
            raise ValueError("Feature/split parent hash mismatch")
        for part in ("train", "val", "test"):
            if getattr(bundle, f"{part}_cell_ids") != getattr(split, f"{part}_cell_ids"):
                raise ValueError(f"Feature/split cell ordering mismatch: {part}")
        files = [split_file, *(p for p in prep.iterdir() if p.is_file())]
        graph_root = paths.graphs_dir / dataset_name / split_id
        for name in GRAPHS:
            directory = graph_root / name
            graph = GraphBundle.load(directory)
            ids = bundle.train_cell_ids + bundle.val_cell_ids + bundle.test_cell_ids
            sections = bundle.section_ids_train + bundle.section_ids_val + bundle.section_ids_test
            if graph.node_cell_ids != ids or graph.node_section_ids != sections:
                raise ValueError(f"Graph node/section ordering mismatch: {name}")
            if graph.manifest.feature_manifest_hash != bundle.manifest.compute_manifest_hash():
                raise ValueError(f"Graph/feature parent mismatch: {name}")
            if not validate_graph_construction(directory, verbose=False):
                raise ValueError(f"Graph topology invalid: {name}")
            files.extend(p for p in directory.iterdir() if p.is_file())
        if (paths.root_dir / INDEX).is_file():
            verify_paths(paths.root_dir, read_index(paths.root_dir), files)
        print(f"PASS: {dataset_name}/{split_id} actual bytes, parents, order and topology")
        return True
    except (ValueError, OSError, KeyError) as error:
        print(f"FAIL: {dataset_name}/{split_id}: {error}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if validate_all_artifacts(args.dataset, args.split) else 1)
