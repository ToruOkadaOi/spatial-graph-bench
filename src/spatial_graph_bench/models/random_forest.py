"""Spatially ignorant Random Forest baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from spatial_graph_bench.config.model import RandomForestConfig


class RandomForestBaseline:
    """Random Forest tabular baseline."""

    def __init__(self, config: RandomForestConfig | None = None, random_state: int = 42) -> None:
        self.config = config or RandomForestConfig()
        self.clf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            n_jobs=self.config.n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray, num_classes: int | None = None) -> np.ndarray:
        raw_probs = self.clf.predict_proba(X)
        if num_classes is None or raw_probs.shape[1] == num_classes:
            return raw_probs

        full_probs = np.zeros((X.shape[0], num_classes), dtype=np.float32)
        for col_idx, cls_idx in enumerate(self.clf.classes_):
            if int(cls_idx) < num_classes:
                full_probs[:, int(cls_idx)] = raw_probs[:, col_idx]
        return full_probs
