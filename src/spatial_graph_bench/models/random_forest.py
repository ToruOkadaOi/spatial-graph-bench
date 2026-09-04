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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)
