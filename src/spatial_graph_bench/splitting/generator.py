"""Split generator implementing donor, section, replicate, and spatial block partitioning."""

from __future__ import annotations

import pandas as pd

from spatial_graph_bench.config.split import SplitConfig
from spatial_graph_bench.splitting.schema import SplitDefinition
from spatial_graph_bench.utils.logging import get_logger

logger = get_logger("splitting.generator")


def create_split_definition(
    obs: pd.DataFrame,
    config: SplitConfig,
    cell_id_col: str | None = None,
    label_col: str = "cell_type",
) -> SplitDefinition:
    """Construct a frozen SplitDefinition adhering strictly to §3.4 label-space intersection rules."""
    group_col = config.group_column
    if group_col not in obs.columns:
        raise KeyError(
            f"Group column '{group_col}' not found in obs. Available: {list(obs.columns)}"
        )

    cell_ids = (
        obs[cell_id_col].astype(str).tolist() if cell_id_col else [str(idx) for idx in obs.index]
    )
    obs = obs.copy()
    obs["_cell_id"] = cell_ids

    train_mask = obs[group_col].astype(str).isin(config.train_groups)
    val_mask = obs[group_col].astype(str).isin(config.val_groups)
    test_mask = obs[group_col].astype(str).isin(config.test_groups)

    # Internal validation carve if val_groups is empty
    if not any(val_mask) and config.val_ratio_within_train > 0:
        train_indices = (
            obs[train_mask]
            .sample(
                frac=1.0 - config.val_ratio_within_train,
                random_state=config.seed,
            )
            .index
        )
        val_indices = obs[train_mask].index.difference(train_indices)
        train_mask = obs.index.isin(train_indices)
        val_mask = obs.index.isin(val_indices)

    train_cells = obs.loc[train_mask, "_cell_id"].tolist()
    val_cells = obs.loc[val_mask, "_cell_id"].tolist()
    test_cells = obs.loc[test_mask, "_cell_id"].tolist()

    # §3.4 Label space intersection analysis
    if label_col in obs.columns:
        train_labels = set(obs.loc[train_mask, label_col].dropna().astype(str).unique())
        test_labels = set(obs.loc[test_mask, label_col].dropna().astype(str).unique())

        evaluated_labels = sorted(train_labels & test_labels)
        excluded_labels = sorted(test_labels - train_labels)
        coverage = len(evaluated_labels) / max(len(test_labels), 1)

        label_counts: dict[str, dict[str, int]] = {}
        for lab in sorted(train_labels | test_labels):
            label_counts[lab] = {
                "train": int((obs.loc[train_mask, label_col] == lab).sum()),
                "val": int((obs.loc[val_mask, label_col] == lab).sum()),
                "test": int((obs.loc[test_mask, label_col] == lab).sum()),
            }
    else:
        evaluated_labels = []
        excluded_labels = []
        coverage = 1.0
        label_counts = {}

    # Group counts
    group_counts: dict[str, dict[str, int]] = {}
    for grp in obs[group_col].astype(str).unique():
        group_counts[grp] = {
            "train": int((train_mask & (obs[group_col] == grp)).sum()),
            "val": int((val_mask & (obs[group_col] == grp)).sum()),
            "test": int((test_mask & (obs[group_col] == grp)).sum()),
        }

    split = SplitDefinition(
        dataset_name=config.dataset_name,
        split_id=config.split_id,
        hierarchy=config.hierarchy,
        seed=config.seed,
        train_groups=config.train_groups,
        val_groups=config.val_groups,
        test_groups=config.test_groups,
        train_cell_ids=train_cells,
        val_cell_ids=val_cells,
        test_cell_ids=test_cells,
        total_cells=len(obs),
        group_counts=group_counts,
        label_support=label_counts,
        evaluated_labels=evaluated_labels,
        excluded_labels=excluded_labels,
        coverage_ratio=coverage,
        config_hash=config.compute_config_hash(),
    )
    split.validate_disjointness()
    return split
