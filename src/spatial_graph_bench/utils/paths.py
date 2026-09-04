"""Path management and standard directory resolution for spatial-graph-bench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Find the root directory of the spatial-graph-bench repository."""
    if env_root := os.getenv("SPATIAL_GRAPH_BENCH_ROOT"):
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    # Go up 4 levels: utils -> spatial_graph_bench -> src -> repo root
    return current.parents[3]


@dataclass(frozen=True)
class ArtifactPaths:
    """Structured path resolver for project artifacts."""

    root_dir: Path

    @classmethod
    def default(cls) -> ArtifactPaths:
        return cls(root_dir=get_project_root())

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.root_dir / "splits"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def preprocessed_dir(self) -> Path:
        return self.artifacts_dir / "preprocessed"

    @property
    def graphs_dir(self) -> Path:
        return self.artifacts_dir / "graphs"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def audits_dir(self) -> Path:
        return self.root_dir / "audits"

    @property
    def configs_dir(self) -> Path:
        return self.root_dir / "configs"

    def dataset_split_file(self, dataset_name: str, split_id: str) -> Path:
        return self.splits_dir / dataset_name / f"{split_id}.json"

    def dataset_preprocessed_dir(self, dataset_name: str, split_id: str) -> Path:
        return self.preprocessed_dir / dataset_name / split_id

    def dataset_graph_dir(self, dataset_name: str, split_id: str, graph_name: str) -> Path:
        return self.graphs_dir / dataset_name / split_id / graph_name

    def dataset_results_dir(self, dataset_name: str, split_id: str) -> Path:
        return self.results_dir / dataset_name / split_id
