"""Generate frozen canonical and exploratory split files for benchmark datasets."""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd

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


def generate_stereoseq_splits() -> None:
    paths = ArtifactPaths.default()
    p44 = paths.raw_data_dir / "axolotl_telencephalon" / "Stage44.h5ad"
    p54 = paths.raw_data_dir / "axolotl_telencephalon" / "Stage54.h5ad"
    if not p44.is_file() or not p54.is_file():
        raise FileNotFoundError(f"Stereo-seq raw files missing: {p44} or {p54}")

    logger.info("Loading Stereo-seq raw files: %s and %s", p44, p54)
    a44 = ad.read_h5ad(p44, backed="r")
    a54 = ad.read_h5ad(p54, backed="r")

    obs44 = a44.obs.copy()
    obs54 = a54.obs.copy()

    obs44["slice"] = "Stage44"
    obs54["slice"] = "Stage54"

    obs44["global_cell_id"] = ["Stage44_" + str(idx) for idx in obs44.index]
    obs54["global_cell_id"] = ["Stage54_" + str(idx) for idx in obs54.index]

    obs44.index = obs44["global_cell_id"]
    obs54.index = obs54["global_cell_id"]

    joint_obs = pd.concat([obs54, obs44])

    canonical_cfg = SplitConfig(
        dataset_name="stereoseq_axolotl_telencephalon",
        split_id="replicate_held_out_canonical",
        hierarchy=SplitHierarchy.REPLICATE_HELD_OUT,
        seed=42,
        train_groups=["Stage54"],
        val_groups=[],
        val_ratio_within_train=0.2,
        test_groups=["Stage44"],
        group_column="slice",
    )

    canonical_split = create_split_definition(
        joint_obs,
        canonical_cfg,
        cell_id_col="global_cell_id",
        label_col="Annotation",
    )
    out_canonical = paths.dataset_split_file(
        "stereoseq_axolotl_telencephalon", "replicate_held_out_canonical"
    )
    canonical_split.save_json(out_canonical)
    logger.info("Saved Stereo-seq canonical split: %s", out_canonical)

    p57 = paths.raw_data_dir / "axolotl_telencephalon" / "Stage57.h5ad"
    if p57.is_file():
        logger.info("Loading Stereo-seq Stage57: %s", p57)
        a57 = ad.read_h5ad(p57, backed="r")
        obs57 = a57.obs.copy()
        obs57["slice"] = "Stage57"
        obs57["global_cell_id"] = ["Stage57_" + str(idx) for idx in obs57.index]
        obs57.index = obs57["global_cell_id"]
        joint_obs_3stage = pd.concat([obs54, obs44, obs57])

        three_stage_cfg = SplitConfig(
            dataset_name="stereoseq_axolotl_telencephalon",
            split_id="developmental_three_stage",
            hierarchy=SplitHierarchy.REPLICATE_HELD_OUT,
            seed=42,
            train_groups=["Stage44", "Stage54"],
            val_groups=[],
            val_ratio_within_train=0.2,
            test_groups=["Stage57"],
            group_column="slice",
        )
        three_stage_split = create_split_definition(
            joint_obs_3stage,
            three_stage_cfg,
            cell_id_col="global_cell_id",
            label_col="Annotation",
        )
        out_three_stage = paths.dataset_split_file(
            "stereoseq_axolotl_telencephalon", "developmental_three_stage"
        )
        three_stage_split.save_json(out_three_stage)
        logger.info("Saved Stereo-seq 3-stage split: %s", out_three_stage)


def generate_openst_splits() -> None:
    paths = ArtifactPaths.default()
    p3d = paths.raw_data_dir / "openst_human_lymph_node" / "GSE251926_metastatic_lymph_node_3d.h5ad"
    if not p3d.is_file():
        raise FileNotFoundError(f"Open-ST 3D raw file missing: {p3d}")

    logger.info("Loading Open-ST 3D dataset: %s", p3d)
    a = ad.read_h5ad(p3d, backed="r")
    obs = a.obs[["n_section", "annotation"]].copy()

    # Filter strictly to Section 6 (reference) and Section 19 (query)
    obs = obs[obs["n_section"].isin([6, 19])].copy()
    obs = obs[obs["annotation"] != "unknown"].copy()

    # Ensure shared classes
    s6_classes = set(obs[obs["n_section"] == 6]["annotation"].value_counts().index)
    s19_classes = set(obs[obs["n_section"] == 19]["annotation"].value_counts().index)
    shared_classes = sorted(s6_classes & s19_classes)
    obs = obs[obs["annotation"].isin(shared_classes)].copy()
    obs["section_str"] = obs["n_section"].astype(str)
    obs["cell_id"] = obs.index.astype(str)

    canonical_cfg = SplitConfig(
        dataset_name="openst_human_lymph_node",
        split_id="section_held_out_canonical",
        hierarchy=SplitHierarchy.SECTION_HELD_OUT,
        seed=42,
        train_groups=["6"],
        val_groups=[],
        val_ratio_within_train=0.2,
        test_groups=["19"],
        group_column="section_str",
    )

    canonical_split = create_split_definition(
        obs,
        canonical_cfg,
        cell_id_col="cell_id",
        label_col="annotation",
    )
    out_canonical = paths.dataset_split_file(
        "openst_human_lymph_node", "section_held_out_canonical"
    )
    canonical_split.save_json(out_canonical)
    logger.info("Saved Open-ST canonical split: %s", out_canonical)


def generate_xenium_splits() -> None:
    paths = ArtifactPaths.default()
    h5_path = paths.raw_data_dir / "xenium_mouse_kidney" / "Xenium.h5ad"
    if not h5_path.is_file():
        raise FileNotFoundError(f"Xenium raw file missing: {h5_path}")

    logger.info("Loading Xenium dataset: %s", h5_path)
    adata = ad.read_h5ad(h5_path, backed="r")
    obs = adata.obs[adata.obs["ident"].isin(["ShamL", "ShamR"])][["ident", "celltype_plot"]].copy()
    obs["cell_id"] = obs.index.astype(str)

    canonical_cfg = SplitConfig(
        dataset_name="xenium_mouse_kidney",
        split_id="replicate_held_out_canonical",
        hierarchy=SplitHierarchy.REPLICATE_HELD_OUT,
        seed=42,
        train_groups=["ShamL"],
        val_groups=[],
        val_ratio_within_train=0.2,
        test_groups=["ShamR"],
        group_column="ident",
    )

    canonical_split = create_split_definition(
        obs,
        canonical_cfg,
        cell_id_col="cell_id",
        label_col="celltype_plot",
    )
    out_canonical = paths.dataset_split_file(
        "xenium_mouse_kidney", "replicate_held_out_canonical"
    )
    canonical_split.save_json(out_canonical)
    logger.info("Saved Xenium canonical split: %s", out_canonical)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    args = parser.parse_args()

    if args.dataset == "merfish_mouse_spinal_cord":
        generate_merfish_splits()
    elif args.dataset == "stereoseq_axolotl_telencephalon":
        generate_stereoseq_splits()
    elif args.dataset == "openst_human_lymph_node":
        generate_openst_splits()
    elif args.dataset == "xenium_mouse_kidney":
        generate_xenium_splits()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
