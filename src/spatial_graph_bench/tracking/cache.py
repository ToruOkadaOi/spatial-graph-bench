"""Cache identity checks; never silently reuse a different experiment."""

from pathlib import Path

from spatial_graph_bench.utils.hashing import hash_dict, hash_file
from spatial_graph_bench.utils.paths import get_project_root


def source_fingerprint() -> str:
    root = get_project_root()
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py")])
    return hash_dict({str(p.relative_to(root)): hash_file(p) for p in paths})


def require_compatible_cache(manifest, config, features, graph, run_dir: Path) -> None:
    expected = {
        "model_config_hash": config.compute_config_hash(),
        "feature_manifest_hash": features.manifest.compute_manifest_hash(),
        "graph_artifact_hash": graph.manifest.compute_manifest_hash(),
        "source_fingerprint": source_fingerprint(),
    }
    mismatches = [key for key, value in expected.items() if getattr(manifest, key, None) != value]
    if mismatches:
        raise ValueError(
            f"Existing run {run_dir} is incompatible or lacks cache provenance: {mismatches}. "
            "Use a separate results root for a new experiment; original results will not be overwritten."
        )
