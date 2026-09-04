"""Generate frozen canonical and exploratory split files for benchmark datasets."""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np

from spatial_graph_bench.config.split import SplitConfig, SplitHierarchy
from spatial_graph_bench.splitting.generator import create_split_definition
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.paths import ArtifactPaths

logger = get_logger("scripts.generate_splits")


def generate_merfish_splits() -> None:
    paths = ArtifactPaths.default()
    raw_path = paths.raw_data_dir / "merfish_spinal_cord" / "MERFISH_spinal_cord_resolved_0718.h5ad"
    if not raw_path.is_file():
        raise FileNotFoundError(f"MERFISH raw file missing: {raw_path}")

    logger.info("Loading MERFISH raw h5ad: %s", raw_path)
    adata = ad.read_h5ad(raw_path, backed="r")

    # Filter to the canonical 18 sections
    sec_list = [
        "0503_F5_T",
        "0503_F5_C",
        "0503_F5_L",
        "0503_F5_S",
        "0503_F4_C",
        "0503_F4_T",
        "0503_F3_C",
        "0503_F4_S",
        "0503_F3_L",
        "0503_F3_T",
        "0503_M5_C",
        "0503_F3_S",
        "0503_M5_T",
        "0503_M4_C",
        "0503_M5_S",
        "0503_M4_L",
        "0503_M4_T",
        "0503_M4_S",
    ]
    obs = adata.obs[adata.obs["Section ID"].isin(sec_list)].copy()
    logger.info("Filtered to 18 sections: %d cells total", len(obs))

    # 1. Canonical Donor/Mouse-Held-Out Split
    canonical_cfg = SplitConfig(
        dataset_name="merfish_mouse_spinal_cord",
        split_id="mouse_held_out_canonical",
        hierarchy=SplitHierarchy.DONOR_HELD_OUT,
        seed=42,
        train_groups=["F3", "M4", "F5"],
        val_groups=["M5"],
        test_groups=["F4"],
        group_column="Mouse ID",
    )
    canonical_split = create_split_definition(
        obs,
        canonical_cfg,
        label_col="MERFISH cell type annotation",
    )
    out_canonical = paths.dataset_split_file(
        "merfish_mouse_spinal_cord", "mouse_held_out_canonical"
    )
    canonical_split.save_json(out_canonical)
    logger.info("Saved canonical split: %s", out_canonical)

    # 2. Exploratory Spatial-Block Split (Quadrant holdout on F3 cervical section)
    f3_cervical = obs[obs["Section ID"] == "0503_F3_C"].copy()
    cx = f3_cervical["center_x"].to_numpy()
    cy = f3_cervical["center_y"].to_numpy()
    mx = float(np.median(cx))
    my = float(np.median(cy))

    # Assign quadrants
    quads = []
    for x, y in zip(cx, cy, strict=True):
        if x >= mx and y >= my:
            quads.append("Q1_NE")
        elif x < mx and y >= my:
            quads.append("Q2_NW")
        elif x < mx and y < my:
            quads.append("Q3_SW")
        else:
            quads.append("Q4_SE")
    f3_cervical["quadrant"] = quads

    block_cfg = SplitConfig(
        dataset_name="merfish_mouse_spinal_cord",
        split_id="spatial_block_exploratory",
        hierarchy=SplitHierarchy.SPATIAL_BLOCK,
        seed=42,
        train_groups=["Q1_NE", "Q2_NW"],
        val_groups=["Q3_SW"],
        test_groups=["Q4_SE"],
        group_column="quadrant",
    )
    block_split = create_split_definition(
        f3_cervical,
        block_cfg,
        label_col="MERFISH cell type annotation",
    )
    out_block = paths.dataset_split_file("merfish_mouse_spinal_cord", "spatial_block_exploratory")
    block_split.save_json(out_block)
    logger.info("Saved exploratory spatial-block split: %s", out_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    args = parser.parse_args()

    if args.dataset == "merfish_mouse_spinal_cord":
        generate_merfish_splits()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
