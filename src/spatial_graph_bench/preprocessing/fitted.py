"""Portable, pickle-free fitted preprocessing parameters for inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedPreprocessor:
    """The train-fitted feature transformation used to produce PCA inputs."""

    input_feature_names: list[str]
    selected_feature_indices: np.ndarray
    normalize_library_size: bool
    target_sum: float
    scale_features: bool
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pipeline_version: str

    def transform(
        self,
        counts: np.ndarray | sparse.spmatrix,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Transform counts using the saved gene order and train-fitted parameters."""
        if feature_names is not None and list(feature_names) != self.input_feature_names:
            raise ValueError("Input feature names/order differ from the fitted preprocessing panel")
        if counts.ndim != 2 or counts.shape[1] != len(self.input_feature_names):
            raise ValueError(
                f"Expected a cell-by-feature matrix with {len(self.input_feature_names)} columns"
            )
        X = counts.tocsr(copy=True) if sparse.issparse(counts) else sparse.csr_matrix(counts)
        if self.normalize_library_size:
            totals = np.asarray(X.sum(axis=1)).ravel()
            totals[totals == 0] = 1.0
            X = X.multiply((self.target_sum / totals)[:, None]).tocsr()
            X.data = np.log1p(X.data)
        X = X[:, self.selected_feature_indices].toarray()
        if self.scale_features:
            if self.scaler_mean is None or self.scaler_scale is None:
                raise ValueError("Saved scaler parameters are incomplete")
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.mean_ = self.scaler_mean
            scaler.scale_ = self.scaler_scale
            scaler.var_ = np.ones_like(self.scaler_scale)
            scaler.n_features_in_ = len(self.selected_feature_indices)
            scaler.n_samples_seen_ = 1
            X = scaler.transform(X)
        projected = X @ self.pca_components.T
        projected -= self.pca_mean @ self.pca_components.T
        return np.ascontiguousarray(projected, dtype=np.float32)

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "selected_feature_indices": np.asarray(self.selected_feature_indices, dtype=np.int64),
            "pca_mean": np.asarray(self.pca_mean),
            "pca_components": np.asarray(self.pca_components),
        }
        if self.scaler_mean is not None:
            arrays["scaler_mean"] = np.asarray(self.scaler_mean)
        if self.scaler_scale is not None:
            arrays["scaler_scale"] = np.asarray(self.scaler_scale)
        np.savez_compressed(directory / "preprocessing_parameters.npz", **arrays)
        metadata = {
            "format_version": 1,
            "pipeline_version": self.pipeline_version,
            "input_feature_names": self.input_feature_names,
            "normalize_library_size": self.normalize_library_size,
            "target_sum": self.target_sum,
            "scale_features": self.scale_features,
            "parameters_file": "preprocessing_parameters.npz",
        }
        (directory / "preprocessing_model.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> FittedPreprocessor:
        directory = Path(directory)
        metadata = json.loads((directory / "preprocessing_model.json").read_text())
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported fitted-preprocessor format")
        with np.load(directory / metadata["parameters_file"], allow_pickle=False) as parameters:
            return cls(
                input_feature_names=metadata["input_feature_names"],
                selected_feature_indices=parameters["selected_feature_indices"],
                normalize_library_size=metadata["normalize_library_size"],
                target_sum=metadata["target_sum"],
                scale_features=metadata["scale_features"],
                scaler_mean=parameters.get("scaler_mean", None),
                scaler_scale=parameters.get("scaler_scale", None),
                pca_mean=parameters["pca_mean"],
                pca_components=parameters["pca_components"],
                pipeline_version=metadata["pipeline_version"],
            )
