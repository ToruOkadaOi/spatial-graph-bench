"""Create a new explicit byte anchor; never run automatically during validation.

This is a migration aid for maintainers, not a way to repair failed validation.
Review the resulting index against release archives before accepting it in Git.
"""

import json
from pathlib import Path

import numpy as np

from spatial_graph_bench.analysis.catalog import DATASETS
from spatial_graph_bench.analysis.integrity import INDEX
from spatial_graph_bench.utils.hashing import hash_array, hash_dict, hash_file
from spatial_graph_bench.utils.paths import get_project_root


def seal(root: Path) -> None:
    target = root / INDEX
    if target.exists():
        raise FileExistsError("Existing v2 anchor is immutable; use a new migration version")
    files, discrepancies = {}, {}
    for ds, sp, _ in DATASETS:
        prep = root / "artifacts/preprocessed" / ds / sp
        manifest = json.loads((prep / "feature_manifest.json").read_text())
        discrepancies[f"{ds}/{sp}"] = {
            part: {
                "claimed": manifest[f"pca_{part}_hash"],
                "actual": hash_array(np.load(prep / f"X_pca_{part}.npy")),
            }
            for part in ("train", "val", "test")
            if manifest[f"pca_{part}_hash"] != hash_array(np.load(prep / f"X_pca_{part}.npy"))
        }
        paths = list(prep.iterdir())
        paths += list((root / "artifacts/graphs" / ds / sp).glob("*/*"))
        results = root / "artifacts/results" / ds / sp
        for name in (
            "run_manifest.json",
            "metrics_summary.json",
            "training_history.csv",
            "test_preds.npy",
            "test_probs.npy",
        ):
            paths += list(results.glob(f"*/{name}"))
        paths.append(root / "splits" / ds / f"{sp}.json")
        for file in sorted(paths):
            if file.is_file():
                files[file.relative_to(root).as_posix()] = hash_file(file)
    data = {
        "schema_version": 2,
        "migration": "2026-10-25 historical byte anchor; original manifests unchanged",
        "legacy_pca_hash_discrepancies": discrepancies,
        "files": dict(sorted(files.items())),
        "content_hash": hash_dict(files),
    }
    target.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Anchored {len(files)} original files in {target}")


if __name__ == "__main__":
    seal(get_project_root())
