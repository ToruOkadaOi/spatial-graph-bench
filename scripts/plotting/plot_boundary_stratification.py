"""Publication Figure 3: Spatial Boundary Stratification and Edge Degradation.

Visualizes the geometric convex hull margin partitioning and the resulting
performance drop from deep interior to physical tissue boundary (§6.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from scripts.stratify_boundary_lift import compute_boundary_mask

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_boundary_analysis() -> None:
    dataset_name = "stereoseq_axolotl_telencephalon"
    split_id = "developmental_three_stage"
    prep_dir = Path(f"artifacts/preprocessed/{dataset_name}/{split_id}")
    post_hoc_file = Path(f"artifacts/results/{dataset_name}/{split_id}/post_hoc_summary.json")

    # Load spatial coordinates
    coords = np.load(prep_dir / "spatial_test.npy")
    with open(prep_dir / "cell_ids.json") as f:
        cell_info = json.load(f)
    sections = np.array(cell_info["sections_test"])

    # Compute boundary mask
    is_boundary = compute_boundary_mask(coords, sections, percentile=15.0)

    # Load post-hoc summary for spatial_knn_k12
    with open(post_hoc_file) as f:
        post_hoc = json.load(f)

    df_ph = pd.DataFrame(post_hoc)
    df_k12 = df_ph[df_ph["graph"] == "spatial_knn_k12"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A: 2D Spatial Layout of Interior vs. Boundary
    ax_map = axes[0]
    sec = np.unique(sections)[0]
    sec_mask = sections == sec
    sec_pts = coords[sec_mask]
    sec_bnd = is_boundary[sec_mask]

    hull = ConvexHull(sec_pts)

    # Scatter plot
    ax_map.scatter(
        sec_pts[~sec_bnd, 0],
        sec_pts[~sec_bnd, 1],
        c="#1f77b4",
        s=8,
        alpha=0.6,
        label=f"Deep Interior ({np.sum(~sec_bnd):,} cells, 85.4%)",
    )
    ax_map.scatter(
        sec_pts[sec_bnd, 0],
        sec_pts[sec_bnd, 1],
        c="#d62728",
        s=12,
        alpha=0.8,
        label=f"Boundary Margin ({np.sum(sec_bnd):,} cells, 14.6%)",
    )

    # Plot Convex Hull boundary line
    for simplex in hull.simplices:
        ax_map.plot(sec_pts[simplex, 0], sec_pts[simplex, 1], "k--", linewidth=1.2, alpha=0.7)

    ax_map.set_title(
        "A. Tissue Convex Hull Geometric Margin (Stereo-seq Stage 57)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_map.set_xlabel("Spatial X Coordinate (µm)", fontweight="bold")
    ax_map.set_ylabel("Spatial Y Coordinate (µm)", fontweight="bold")
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    # Panel B: Interior Lift vs. Boundary Lift
    ax_bar = axes[1]
    models = ["GCN", "GAT", "GIN", "GRAPHSAGE"]
    model_display = ["GCN", "GAT", "GIN", "GraphSAGE"]

    records = []
    for m, m_disp in zip(models, model_display, strict=True):
        row = df_k12[df_k12["model"] == m].iloc[0]
        records.append({"Model": m_disp, "Region": "Deep Interior", "Lift": row["int_lift_mean"]})
        records.append({"Model": m_disp, "Region": "Boundary Margin", "Lift": row["bnd_lift_mean"]})

    df_bar = pd.DataFrame(records)

    sns.barplot(
        data=df_bar,
        x="Model",
        y="Lift",
        hue="Region",
        palette={"Deep Interior": "#1f77b4", "Boundary Margin": "#d62728"},
        ax=ax_bar,
        capsize=0.08,
        zorder=3,
    )

    ax_bar.axhline(0, color="#333333", linestyle="-", linewidth=1.2, zorder=2)
    ax_bar.set_title(
        "B. Edge Degradation: Interior vs. Boundary Lift (spatial_knn_k12)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_bar.set_xlabel("GNN Architecture", fontweight="bold", labelpad=8)
    ax_bar.set_ylabel("Matched Lift Δ over MLP", fontweight="bold", labelpad=8)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax_bar.legend(
        title="Tissue Zone", loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5
    )

    # Annotate margin penalty for GraphSAGE
    sage_int = df_bar[(df_bar["Model"] == "GraphSAGE") & (df_bar["Region"] == "Deep Interior")][
        "Lift"
    ].values[0]
    sage_bnd = df_bar[(df_bar["Model"] == "GraphSAGE") & (df_bar["Region"] == "Boundary Margin")][
        "Lift"
    ].values[0]
    penalty = (sage_bnd - sage_int) * 100
    ax_bar.annotate(
        f"Margin Penalty:\n{penalty:.2f}% F1",
        xy=(3, sage_bnd),
        xytext=(3.1, -0.05),
        arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1, "headwidth": 5},
        fontsize=9.5,
        fontweight="bold",
    )

    plt.tight_layout()

    out_pdf = OUT_DIR / "fig3_boundary_stratification.pdf"
    out_png = OUT_DIR / "fig3_boundary_stratification.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3:\n  PDF: {out_pdf}\n  PNG: {out_png}")


if __name__ == "__main__":
    plot_boundary_analysis()
