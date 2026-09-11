"""Publication Figure 4: Tubular-Interstitial Feature Over-Smoothing in Xenium Mouse Kidney.

Dissects the cellular and architectural mechanism behind isotropic GNN collapse:
1. Panel A: Spatial tissue architecture of the test kidney section (ShamR) separating
   dense tubular epithelium from sparse interstitial/endothelial/immune niches.
2. Panel B: Per-class matched lift comparison (GCN catastrophic collapse vs. GraphSAGE resilience).
3. Panel C: Misclassification shift (Delta C = C_GCN - C_MLP) demonstrating how interstitial
   and endothelial cells are falsely absorbed into predominant tubular epithelial predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.05)
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


def plot_kidney_interstitial_oversmoothing() -> None:
    dataset_name = "xenium_mouse_kidney"
    split_id = "replicate_held_out_canonical"
    prep_dir = Path(f"artifacts/preprocessed/{dataset_name}/{split_id}")
    res_dir = Path(f"artifacts/results/{dataset_name}/{split_id}")

    # Load spatial coordinates and labels
    coords = np.load(prep_dir / "spatial_test.npy")
    y_true = np.load(prep_dir / "test_labels.npy")

    with open(prep_dir / "label_mapping.json") as f:
        l2i = json.load(f)

    # Load per-class lift summaries
    with open(res_dir / "per_class_lift_gcn_spatial_knn_k12.json") as f:
        gcn_data = json.load(f)
    with open(res_dir / "per_class_lift_graphsage_spatial_knn_k12.json") as f:
        sage_data = json.load(f)

    # 3-Panel Figure
    fig = plt.figure(figsize=(22, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.28)

    # -------------------------------------------------------------
    # Panel A: Spatial Compartment Map of Held-out Kidney Replicate
    # -------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])

    pt_ids = {l2i[k] for k in ["PTS1", "PTS2", "PTS3"] if k in l2i}
    dt_ids = {l2i[k] for k in ["TAL", "DCT", "CNT", "PC", "ICB", "ICA", "DTL"] if k in l2i}
    inter_ids = {l2i[k] for k in ["EC", "Fib", "Per-SMC"] if k in l2i}
    glom_ids = {l2i[k] for k in ["Pod", "Glom-EC", "PEC"] if k in l2i}
    immune_ids = {l2i[k] for k in ["Immune", "Uro"] if k in l2i}

    rng = np.random.default_rng(42)
    n_pts = len(coords)
    sub_idx = rng.choice(n_pts, size=min(35000, n_pts), replace=False)
    sub_coords = coords[sub_idx]
    sub_y = y_true[sub_idx]

    c_pt = np.isin(sub_y, list(pt_ids))
    c_dt = np.isin(sub_y, list(dt_ids))
    c_inter = np.isin(sub_y, list(inter_ids))
    c_glom = np.isin(sub_y, list(glom_ids))
    c_imm = np.isin(sub_y, list(immune_ids))

    ax0.scatter(
        sub_coords[c_pt, 0],
        sub_coords[c_pt, 1],
        c="#3498db",
        s=1.5,
        alpha=0.45,
        label="Proximal Tubules (44.6%)",
        rasterized=True,
    )
    ax0.scatter(
        sub_coords[c_dt, 0],
        sub_coords[c_dt, 1],
        c="#9b59b6",
        s=1.5,
        alpha=0.45,
        label="Distal / Loop / Collecting (28.4%)",
        rasterized=True,
    )
    ax0.scatter(
        sub_coords[c_inter, 0],
        sub_coords[c_inter, 1],
        c="#e67e22",
        s=3.0,
        alpha=0.75,
        label="Interstitial / Endothelial (17.5%)",
        rasterized=True,
    )
    ax0.scatter(
        sub_coords[c_glom, 0],
        sub_coords[c_glom, 1],
        c="#e74c3c",
        s=6.0,
        alpha=0.9,
        label="Glomerular Structures (4.8%)",
        rasterized=True,
    )
    ax0.scatter(
        sub_coords[c_imm, 0],
        sub_coords[c_imm, 1],
        c="#2ecc71",
        s=4.0,
        alpha=0.8,
        label="Immune / Urothelial (4.0%)",
        rasterized=True,
    )

    ax0.set_title("A. Tissue Architecture (Test Kidney Replicate)", fontweight="bold", pad=12)
    ax0.set_xlabel("Spatial X (µm)", fontweight="bold")
    ax0.set_ylabel("Spatial Y (µm)", fontweight="bold")
    ax0.set_aspect("equal")
    ax0.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.5, markerscale=2.5)

    # -------------------------------------------------------------
    # Panel B: Per-Class Matched Lift Contrast (GCN vs. GraphSAGE)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1])

    df_gcn = pd.DataFrame(gcn_data["classes"])
    df_sage = pd.DataFrame(sage_data["classes"])

    df_gcn["Model"] = "GCN (k=12)"
    df_sage["Model"] = "GraphSAGE (k=12)"

    df_plot = pd.concat([df_gcn, df_sage], ignore_index=True)
    order = df_gcn.sort_values("lift_mean")["class_name"].tolist()

    palette = {"GCN (k=12)": "#e74c3c", "GraphSAGE (k=12)": "#2ecc71"}
    sns.barplot(
        data=df_plot,
        x="class_name",
        y="lift_mean",
        hue="Model",
        order=order,
        palette=palette,
        ax=ax1,
        edgecolor="#222222",
        linewidth=0.6,
    )

    eps = 0.0241
    ax1.axhspan(-eps, eps, color="#999999", alpha=0.2, label=f"Parity Margin (±{eps:.3f})")
    ax1.axhline(0, color="#444444", linestyle="--", linewidth=1.2)

    ax1.set_title("B. Per-Class Matched Lift (Spatial k=12 vs. MLP)", fontweight="bold", pad=12)
    ax1.set_xlabel("Cell Type (Ordered by GCN Degradation)", fontweight="bold")
    ax1.set_ylabel("Matched Lift Δ Macro-F1", fontweight="bold")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel C: Interstitial Drowning in Tubular Epithelium
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2])

    mlp_preds = np.load(res_dir / "mlp_none_seed42" / "test_preds.npy")
    gcn_preds = np.load(res_dir / "gcn_spatial_knn_k12_seed42" / "test_preds.npy")

    focus_types = ["Immune", "Fib", "EC", "Glom-EC", "PEC"]
    target_types = ["PTS1", "PTS2", "PTS3", "TAL", "DCT"]

    focus_ids = [l2i[t] for t in focus_types if t in l2i]
    target_ids = [l2i[t] for t in target_types if t in l2i]

    delta_matrix = np.zeros((len(focus_types), len(target_types)))

    for i, f_id in enumerate(focus_ids):
        mask = y_true == f_id
        n_cells = np.sum(mask)
        if n_cells == 0:
            continue
        mlp_pred_counts = np.bincount(mlp_preds[mask], minlength=len(l2i))
        gcn_pred_counts = np.bincount(gcn_preds[mask], minlength=len(l2i))

        for j, t_id in enumerate(target_ids):
            frac_mlp = mlp_pred_counts[t_id] / n_cells
            frac_gcn = gcn_pred_counts[t_id] / n_cells
            delta_matrix[i, j] = (frac_gcn - frac_mlp) * 100.0

    sns.heatmap(
        delta_matrix,
        xticklabels=target_types,
        yticklabels=focus_types,
        cmap="Reds",
        annot=True,
        fmt="+.1f",
        cbar_kws={"label": "Misclassification Shift Δ (%) (GCN - MLP)"},
        ax=ax2,
        linewidths=0.5,
        linecolor="#dddddd",
    )

    ax2.set_title("C. Interstitial False Assimilation into Tubules", fontweight="bold", pad=12)
    ax2.set_xlabel("Falsely Predicted Tubular Epithelium", fontweight="bold")
    ax2.set_ylabel("True Interstitial / Glomerular Identity", fontweight="bold")

    plt.tight_layout()

    out_pdf = OUT_DIR / "fig4_kidney_interstitial_oversmoothing.pdf"
    out_png = OUT_DIR / "fig4_kidney_interstitial_oversmoothing.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 4:\n  PDF: {out_pdf}\n  PNG: {out_png}")


if __name__ == "__main__":
    plot_kidney_interstitial_oversmoothing()
