"""Fetch missing release artifacts, verify archives, and retain existing originals."""

import argparse
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

from spatial_graph_bench.analysis.integrity import read_index, required_for_analysis, verify_index
from spatial_graph_bench.utils.hashing import hash_bytes, hash_file
from spatial_graph_bench.utils.paths import get_project_root


def prepare(root: Path, offline: bool = False) -> None:
    index = read_index(root)
    # Fail before writing anything if an existing immutable file has changed.
    missing = []
    for name, expected in index["files"].items():
        path = root / name
        if not path.exists():
            if required_for_analysis(name):
                missing.append(name)
        elif hash_file(path) != expected:
            raise ValueError(f"Existing historical artifact changed: {name}")
    # Three v1 GPU releases omit baselines; four local Stereo-seq runs also
    # differ from their release. The supplement preserves these exact local bytes.
    with zipfile.ZipFile(root / "audits/reanalysis_recovery_v2.zip") as archive:
        for name in archive.namelist():
            if name not in missing:
                continue
            data = archive.read(name)
            if hash_bytes(data) != index["files"].get(name):
                raise ValueError(f"Baseline supplement bytes mismatch: {name}")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            missing.remove(name)
    assets = json.loads((root / "configs/release_assets.json").read_text())
    for asset in assets:
        prefix = asset["prefix"]
        prefixes = (prefix, asset.get("additional_prefix", prefix))
        if not any(name.startswith(prefixes) for name in missing):
            continue
        archive = root / "dist" / asset["filename"]
        archive.parent.mkdir(exist_ok=True)
        if not archive.is_file():
            if offline:
                raise FileNotFoundError(f"Offline artifact archive missing: {archive}")
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    asset["tag"],
                    "--repo",
                    "ToruOkadaOi/spatial-graph-bench",
                    "--pattern",
                    asset["filename"],
                    "--dir",
                    str(archive.parent),
                ],
                check=True,
            )
        if hash_file(archive) != asset["sha256"]:
            raise ValueError(f"Release archive checksum mismatch: {archive}")
        wanted = {
            name for name in missing if name.startswith(prefixes) and not (root / name).exists()
        }
        # Stream only required files; do not unpack multi-GB probability arrays
        # or create a second complete temporary copy of the archive contents.
        with tarfile.open(archive) as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if asset["kind"] == "results" and not name.startswith("artifacts/"):
                    name = prefix + name
                if name not in wanted or not member.isfile():
                    continue
                data = tar.extractfile(member).read()
                if hash_bytes(data) != index["files"][name]:
                    raise ValueError(f"Release does not contain expected anchored bytes: {name}")
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                wanted.remove(name)
        if wanted:
            raise ValueError(f"Required files missing from release: {sorted(wanted)}")
    verified = verify_index(root)
    print(
        f"Verified {verified['verified_files_count']} historical files; all analysis inputs are present"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(get_project_root(), args.offline)
