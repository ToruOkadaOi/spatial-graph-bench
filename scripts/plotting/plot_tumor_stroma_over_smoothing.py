"""Publication Figure 4: The Tumor-Stroma Boundary Dilution Mechanism in Open-ST.

Dissects the cellular and architectural mechanism behind isotropic GNN collapse:
1. Panel A: Spatial layout of Section 19 showing malignant nests, stromal boundary tracks,
   and focal keratin pearl niches.
2. Panel B: Per-class matched lift contrast (GCN collapse vs. GraphSAGE resilience).
3. Panel C: Misclassification shift (Delta C = C_GCN - C_MLP) showing stromal cells
   absorbed into tumor predictions.
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


def plot_tumor_stroma_over_smoothing() -> None:
    dataset_name = "openst_human_lymph_node"
    split_id = "section_held_out_canonical"
    prep_dir = Path(f"artifacts/preprocessed/{dataset_name}/{split_id}")
    res_dir = Path(f"artifacts/results/{dataset_name}/{split_id}")
    snap_dir = Path(f"audits/baselines_snapshot/{dataset_name}/{split_id}")

    # Load spatial coordinates and labels
    coords = np.load(prep_dir / "spatial_test.npy")
    y_true = np.load(prep_dir / "test_labels.npy")

    with open(prep_dir / "label_mapping.json") as f:
        l2i = json.load(f)

    # Load per-class lift summaries
    with open(res_dir / "per_class_lift_gcn_spatial_knn_k6.json") as f:
        gcn_data = json.load(f)
    with open(res_dir / "per_class_lift_graphsage_spatial_knn_k6.json") as f:
        sage_data = json.load(f)

    # 3-Panel Figure
    fig = plt.figure(figsize=(22, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=0.28)

    # -------------------------------------------------------------
    # PANEL A: Spatial Microenvironment Layout (Section 19)
    # -------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0])

    # Distinct palette for histological components
    color_map = {
        "Tumor": "#e7969c",  # Light coral / malignant field
        "Tumor_Keratin_Pearl": "#d62728",  # Crimson / differentiated focus
        "CAF": "#1f77b4",  # Deep blue / cancer-associated fibroblasts
        "CAM": "#ff7f0e",  # Orange / adhesion stroma
        "High_Endothelial_Venules": "#9467bd",  # Purple / vascular tracks
        "M1_macrophages": "#8c564b",  # Brown / border infiltration
        "T_cell": "#2ca02c",  # Green / lymphoid infiltrate
        "Plasma_IgG": "#aec7e8",  # Light blue / lymphoid background
        "Germinal_Center_Plasma_IgM_B_cell": "#c7c7c7",
    }

    # Plot background lymphoid cells first
    other_mask = ~np.isin(
        y_true,
        [
            l2i["Tumor"],
            l2i["Tumor_Keratin_Pearl"],
            l2i["CAF"],
            l2i["CAM"],
            l2i["High_Endothelial_Venules"],
        ],
    )
    ax_map.scatter(
        coords[other_mask, 0],
        coords[other_mask, 1],
        c="#e0e0e0",
        s=3,
        alpha=0.3,
        label="Lymphoid / Other",
        rasterized=True,
    )

    # Plot Tumor field
    tumor_mask = y_true == l2i["Tumor"]
    ax_map.scatter(
        coords[tumor_mask, 0],
        coords[tumor_mask, 1],
        c=color_map["Tumor"],
        s=5,
        alpha=0.4,
        label=f"Tumor Nests ({np.sum(tumor_mask):,} cells)",
        rasterized=True,
    )

    # Plot CAFs and HEVs (Boundary Stroma)
    caf_mask = y_true == l2i["CAF"]
    ax_map.scatter(
        coords[caf_mask, 0],
        coords[caf_mask, 1],
        c=color_map["CAF"],
        s=12,
        alpha=0.85,
        label=f"CAF Border ({np.sum(caf_mask):,} cells)",
        rasterized=True,
    )

    hev_mask = y_true == l2i["High_Endothelial_Venules"]
    ax_map.scatter(
        coords[hev_mask, 0],
        coords[hev_mask, 1],
        c=color_map["High_Endothelial_Venules"],
        s=10,
        alpha=0.75,
        label=f"HEV Venules ({np.sum(hev_mask):,} cells)",
        rasterized=True,
    )

    # Plot Keratin Pearls (Focal niches)
    kp_mask = y_true == l2i["Tumor_Keratin_Pearl"]
    ax_map.scatter(
        coords[kp_mask, 0],
        coords[kp_mask, 1],
        c=color_map["Tumor_Keratin_Pearl"],
        s=26,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
        label=f"Keratin Pearls ({np.sum(kp_mask):,} cells)",
        zorder=5,
    )

    ax_map.set_title(
        "A. Spatial Tumor-Stroma Microenvironment (Open-ST Sec 19)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_map.set_xlabel("Spatial X (µm)", fontweight="bold")
    ax_map.set_ylabel("Spatial Y (µm)", fontweight="bold")
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5, markerscale=1.8)

    # -------------------------------------------------------------
    # PANEL B: Per-Class Matched Lift Contrast (GCN vs GraphSAGE)
    # -------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[1])

    # Assemble per-class lift dataframe
    gcn_dict = {c["class_name"]: c for c in gcn_data["classes"]}
    sage_dict = {c["class_name"]: c for c in sage_data["classes"]}

    eval_classes = [c["class_name"] for c in gcn_data["classes"] if c["support"] >= 50]
    eval_classes.sort(key=lambda x: gcn_dict[x]["lift_mean"])  # Sort ascending by GCN lift

    records = []
    for c_name in eval_classes:
        records.append(
            {
                "Cell Type": c_name.replace("_", " "),
                "Model": "GCN (Isotropic)",
                "Lift": gcn_dict[c_name]["lift_mean"],
                "Lift_Std": gcn_dict[c_name]["lift_std"],
            }
        )
        records.append(
            {
                "Cell Type": c_name.replace("_", " "),
                "Model": "GraphSAGE (Decoupled)",
                "Lift": sage_dict[c_name]["lift_mean"],
                "Lift_Std": sage_dict[c_name]["lift_std"],
            }
        )

    df_lift = pd.DataFrame(records)

    sns.barplot(
        data=df_lift,
        y="Cell Type",
        x="Lift",
        hue="Model",
        palette={"GCN (Isotropic)": "#d62728", "GraphSAGE (Decoupled)": "#1f77b4"},
        ax=ax_bar,
        capsize=0.15,
        err_kws={"linewidth": 1.1},
        zorder=3,
    )

    ax_bar.axvline(0, color="#333333", linestyle="-", linewidth=1.2, zorder=2)
    ax_bar.axvspan(-0.0054, 0.0054, color="#999999", alpha=0.2, zorder=1, label="MLP Parity Band")
    ax_bar.set_title(
        "B. Per-Class Matched Lift (k=6 Spatial Graph)", fontsize=12, fontweight="bold", pad=10
    )
    ax_bar.set_xlabel("Matched Lift Δ over MLP Baseline (F1)", fontweight="bold", labelpad=8)
    ax_bar.set_ylabel("")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)
    ax_bar.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # PANEL C: Net Misclassification Shift (Delta C = C_GCN - C_MLP)
    # -------------------------------------------------------------
    ax_conf = fig.add_subplot(gs[2])

    # Average confusion matrices across seeds 42..51
    mlp_conf_list = []
    gcn_conf_list = []

    for seed in range(42, 52):
        with open(snap_dir / f"mlp_none_seed{seed}" / "metrics_summary.json") as f:
            mlp_conf_list.append(json.load(f)["test"]["confusion_matrix"])
        with open(res_dir / f"gcn_spatial_knn_k6_seed{seed}" / "metrics_summary.json") as f:
            gcn_conf_list.append(json.load(f)["test"]["confusion_matrix"])

    c_mlp_mean = np.mean(mlp_conf_list, axis=0)
    c_gcn_mean = np.mean(gcn_conf_list, axis=0)
    delta_c = c_gcn_mean - c_mlp_mean

    # Subset to key focus classes: CAM, CAF, HEV, M1, Tumor, Keratin Pearl
    with open(snap_dir / "mlp_none_seed42" / "metrics_summary.json") as f:
        all_eval_labels = json.load(f)["test"]["evaluated_labels"]

    focus_labels = [
        "CAM",
        "CAF",
        "High_Endothelial_Venules",
        "M1_macrophages",
        "Tumor",
        "Tumor_Keratin_Pearl",
    ]
    focus_display = ["CAM", "CAF", "HEV", "M1 Macro", "Tumor", "Keratin Pearl"]
    focus_indices = [all_eval_labels.index(label_name) for label_name in focus_labels]

    sub_delta = delta_c[np.ix_(focus_indices, focus_indices)]

    # Plot heatmap
    sns.heatmap(
        sub_delta,
        xticklabels=focus_display,
        yticklabels=focus_display,
        annot=True,
        fmt="+.0f",
        cmap="vlag",
        center=0,
        cbar_kws={"label": "Net Change in Predictions (GCN - MLP)"},
        ax=ax_conf,
        linewidths=0.5,
        linecolor="#dddddd",
    )

    ax_conf.set_title(
        "C. Error Shift: Stroma Absorption into Tumor", fontsize=12, fontweight="bold", pad=10
    )
    ax_conf.set_xlabel("Predicted Class", fontweight="bold", labelpad=8)
    ax_conf.set_ylabel("True Class", fontweight="bold", labelpad=8)
    ax_conf.set_xticklabels(focus_display, rotation=35, ha="right", fontsize=9)
    ax_conf.set_yticklabels(focus_display, rotation=0, fontsize=9)

    gs.tight_layout(fig)

    out_pdf = OUT_DIR / "fig4_tumor_stroma_over_smoothing.pdf"
    out_png = OUT_DIR / "fig4_tumor_stroma_over_smoothing.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 4:\n  PDF: {out_pdf}\n  PNG: {out_png}")


if __name__ == "__main__":
    plot_tumor_stroma_over_smoothing()
