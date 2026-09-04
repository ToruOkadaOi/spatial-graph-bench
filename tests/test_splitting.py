"""Unit tests for split definitions, disjointness assertion, and §3.4 label coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from spatial_graph_bench.config.split import SplitConfig, SplitHierarchy
from spatial_graph_bench.splitting.generator import create_split_definition
from spatial_graph_bench.splitting.schema import SplitDefinition


def test_split_disjointness_validation():
    # Overlapping groups should raise ValueError
    with pytest.raises(ValueError, match="overlap"):
        SplitDefinition(
            dataset_name="test_ds",
            split_id="s1",
            hierarchy=SplitHierarchy.DONOR_HELD_OUT,
            seed=42,
            train_groups=["d1", "d2"],
            val_groups=["d2"],
            test_groups=["d3"],
            train_cell_ids=["c1", "c2"],
            val_cell_ids=["c3"],
            test_cell_ids=["c4"],
            total_cells=4,
            config_hash="abc",
        )

    # Overlapping cells should raise ValueError
    with pytest.raises(ValueError, match="overlap"):
        SplitDefinition(
            dataset_name="test_ds",
            split_id="s1",
            hierarchy=SplitHierarchy.DONOR_HELD_OUT,
            seed=42,
            train_groups=["d1"],
            val_groups=["d2"],
            test_groups=["d3"],
            train_cell_ids=["c1", "c2"],
            val_cell_ids=["c2", "c3"],
            test_cell_ids=["c4"],
            total_cells=4,
            config_hash="abc",
        )


def test_create_split_with_unseen_classes():
    # Synthetic obs with an unseen class in test partition (§3.4)
    obs = pd.DataFrame(
        {
            "donor": ["d1", "d1", "d2", "d3", "d3"],
            "cell_type": ["T_cell", "B_cell", "T_cell", "T_cell", "Rare_Subtype"],
        },
        index=["c1", "c2", "c3", "c4", "c5"],
    )

    config = SplitConfig(
        dataset_name="dummy",
        split_id="test_split",
        hierarchy=SplitHierarchy.DONOR_HELD_OUT,
        seed=42,
        train_groups=["d1"],
        val_groups=["d2"],
        test_groups=["d3"],
        group_column="donor",
    )

    split = create_split_definition(obs, config, label_col="cell_type")
    assert split.train_cell_ids == ["c1", "c2"]
    assert split.val_cell_ids == ["c3"]
    assert split.test_cell_ids == ["c4", "c5"]

    # §3.4 check: Rare_Subtype is in test but absent from train
    assert "Rare_Subtype" in split.excluded_labels
    assert "T_cell" in split.evaluated_labels
    assert split.coverage_ratio == 0.5  # 1 evaluated of 2 test labels
