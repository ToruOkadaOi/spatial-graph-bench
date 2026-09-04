"""Preprocessed feature bundle schema and serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spatial_graph_bench.utils.hashing import hash_dict


class FeatureManifest(BaseModel):
    """Cryptographic manifest for a frozen preprocessed feature representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    split_id: str
    split_config_hash: str
    preprocessing_version: str  # "strict_version_a" or "platform_default_b"
    n_pca_components: int
    n_hvg: int
    scale_features: bool
    train_cells_count: int
    val_cells_count: int
    test_cells_count: int
    total_cells: int
    num_features: int
    pca_train_hash: str
    pca_val_hash: str
    pca_test_hash: str
    spatial_coords_hash: str
    label_mapping_hash: str
    coordinate_shuffle_passed: bool = True
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_manifest_hash(self) -> str:
        return hash_dict(self.model_dump(mode="json"))


class PreprocessedBundle:
    """In-memory and on-disk container for frozen features and coordinates."""

    def __init__(
        self,
        X_pca_train: np.ndarray,
        X_pca_val: np.ndarray,
        X_pca_test: np.ndarray,
        spatial_train: np.ndarray,
        spatial_val: np.ndarray,
        spatial_test: np.ndarray,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray,
        train_cell_ids: list[str],
        val_cell_ids: list[str],
        test_cell_ids: list[str],
        label_to_id: dict[str, int],
        manifest: FeatureManifest,
        section_ids_train: list[str] | None = None,
        section_ids_val: list[str] | None = None,
        section_ids_test: list[str] | None = None,
    ) -> None:
        self.X_pca_train = np.ascontiguousarray(X_pca_train, dtype=np.float32)
        self.X_pca_val = np.ascontiguousarray(X_pca_val, dtype=np.float32)
        self.X_pca_test = np.ascontiguousarray(X_pca_test, dtype=np.float32)

        self.spatial_train = np.ascontiguousarray(spatial_train, dtype=np.float32)
        self.spatial_val = np.ascontiguousarray(spatial_val, dtype=np.float32)
        self.spatial_test = np.ascontiguousarray(spatial_test, dtype=np.float32)

        self.train_labels = np.asarray(train_labels, dtype=np.int64)
        self.val_labels = np.asarray(val_labels, dtype=np.int64)
        self.test_labels = np.asarray(test_labels, dtype=np.int64)

        self.train_cell_ids = list(train_cell_ids)
        self.val_cell_ids = list(val_cell_ids)
        self.test_cell_ids = list(test_cell_ids)

        self.label_to_id = dict(label_to_id)
        self.manifest = manifest

        self.section_ids_train = section_ids_train or ["unknown"] * len(train_cell_ids)
        self.section_ids_val = section_ids_val or ["unknown"] * len(val_cell_ids)
        self.section_ids_test = section_ids_test or ["unknown"] * len(test_cell_ids)

    def save(self, out_dir: Path | str) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        np.save(d / "X_pca_train.npy", self.X_pca_train)
        np.save(d / "X_pca_val.npy", self.X_pca_val)
        np.save(d / "X_pca_test.npy", self.X_pca_test)

        np.save(d / "spatial_train.npy", self.spatial_train)
        np.save(d / "spatial_val.npy", self.spatial_val)
        np.save(d / "spatial_test.npy", self.spatial_test)

        np.save(d / "train_labels.npy", self.train_labels)
        np.save(d / "val_labels.npy", self.val_labels)
        np.save(d / "test_labels.npy", self.test_labels)

        (d / "cell_ids.json").write_text(
            json.dumps(
                {
                    "train": self.train_cell_ids,
                    "val": self.val_cell_ids,
                    "test": self.test_cell_ids,
                    "sections_train": self.section_ids_train,
                    "sections_val": self.section_ids_val,
                    "sections_test": self.section_ids_test,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        (d / "label_mapping.json").write_text(
            json.dumps(self.label_to_id, indent=2),
            encoding="utf-8",
        )

        (d / "feature_manifest.json").write_text(
            self.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, in_dir: Path | str) -> PreprocessedBundle:
        d = Path(in_dir)
        manifest = FeatureManifest.model_validate_json(
            (d / "feature_manifest.json").read_text(encoding="utf-8")
        )
        cell_ids_data = json.loads((d / "cell_ids.json").read_text(encoding="utf-8"))
        label_map = json.loads((d / "label_mapping.json").read_text(encoding="utf-8"))

        return cls(
            X_pca_train=np.load(d / "X_pca_train.npy"),
            X_pca_val=np.load(d / "X_pca_val.npy"),
            X_pca_test=np.load(d / "X_pca_test.npy"),
            spatial_train=np.load(d / "spatial_train.npy"),
            spatial_val=np.load(d / "spatial_val.npy"),
            spatial_test=np.load(d / "spatial_test.npy"),
            train_labels=np.load(d / "train_labels.npy"),
            val_labels=np.load(d / "val_labels.npy"),
            test_labels=np.load(d / "test_labels.npy"),
            train_cell_ids=cell_ids_data["train"],
            val_cell_ids=cell_ids_data["val"],
            test_cell_ids=cell_ids_data["test"],
            label_to_id=label_map,
            manifest=manifest,
            section_ids_train=cell_ids_data.get("sections_train"),
            section_ids_val=cell_ids_data.get("sections_val"),
            section_ids_test=cell_ids_data.get("sections_test"),
        )
