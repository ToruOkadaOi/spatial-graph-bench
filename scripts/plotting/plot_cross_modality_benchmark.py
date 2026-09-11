"""Publication Figure 1: Cross-Modality Benchmark Matrix.

Compares test Macro-F1 across 4 GNN architectures and 7 graph topologies for
MERFISH (adult mouse spinal cord) vs. Stereo-seq (developing axolotl brain),
overlaying non-spatial MLP parity bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set publication style
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


def plot_cross_modality_matrix() -> None:
    merfish_csv = Path(
        "results/reports/merfish_mouse_spinal_cord_mouse_held_out_canonical_all_runs_summary.csv"
    )
    stereoseq_csv = Path(
        "results/reports/stereoseq_axolotl_telencephalon_developmental_three_stage_all_runs_summary.csv"
    )
    openst_csv = Path(
        "results/reports/openst_human_lymph_node_section_held_out_canonical_all_runs_summary.csv"
    )
    xenium_csv = Path(
        "results/reports/xenium_mouse_kidney_replicate_held_out_canonical_all_runs_summary.csv"
    )

    df_m = pd.read_csv(merfish_csv)
    df_s = pd.read_csv(stereoseq_csv)
    df_o = pd.read_csv(openst_csv)
    df_x = pd.read_csv(xenium_csv)

    df_m["Dataset"] = "MERFISH (Mouse Spinal Cord)"
    df_s["Dataset"] = "Stereo-seq (Axolotl Telencephalon)"
    df_o["Dataset"] = "Open-ST (Human Lymph Node)"
    df_x["Dataset"] = "10x Xenium (Mouse Kidney)"

    # Filter to GNN sweep runs
    df_gnn_m = df_m[df_m["run_type"] == "sweep"].copy()
    df_gnn_s = df_s[df_s["run_type"] == "sweep"].copy()
    df_gnn_o = df_o[df_o["run_type"] == "sweep"].copy()
    df_gnn_x = df_x[df_x["run_type"] == "sweep"].copy()

    # Baselines
    mlp_m_mean = df_m[df_m["model"] == "mlp"]["test_macro_f1"].mean()
    mlp_m_std = df_m[df_m["model"] == "mlp"]["test_macro_f1"].std()
    mlp_s_mean = df_s[df_s["model"] == "mlp"]["test_macro_f1"].mean()
    mlp_s_std = df_s[df_s["model"] == "mlp"]["test_macro_f1"].std()
    mlp_o_mean = df_o[df_o["model"] == "mlp"]["test_macro_f1"].mean()
    mlp_o_std = df_o[df_o["model"] == "mlp"]["test_macro_f1"].std()
    mlp_x_mean = df_x[df_x["model"] == "mlp"]["test_macro_f1"].mean()
    mlp_x_std = df_x[df_x["model"] == "mlp"]["test_macro_f1"].std()

    # Create 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)

    topologies = [
        "spatial_knn_k6",
        "spatial_knn_k12",
        "bipartite_ref_k20",
        "shuffled_spatial_knn_k6",
        "shuffled_spatial_knn_k12",
        "rewired_spatial_knn_k6",
        "rewired_spatial_knn_k12",
    ]
    topo_labels = [
        "Spatial k=6",
        "Spatial k=12",
        "Bipartite Ref",
        "Shuffled k=6",
        "Shuffled k=12",
        "Rewired k=6",
        "Rewired k=12",
    ]
    topo_map = dict(zip(topologies, topo_labels, strict=True))
    palette = sns.color_palette("Set2", n_colors=4)

    panels = [
        (
            axes[0, 0],
            df_gnn_m,
            "A. MERFISH: Adult Spinal Cord (Heterophilic Parity)",
            mlp_m_mean,
            mlp_m_std,
        ),
        (
            axes[0, 1],
            df_gnn_s,
            "B. Stereo-seq: Axolotl Telencephalon (+6.4% Lift)",
            mlp_s_mean,
            mlp_s_std,
        ),
        (
            axes[1, 0],
            df_gnn_o,
            "C. Open-ST: Human Lymph Node (Tumor Over-smoothing)",
            mlp_o_mean,
            mlp_o_std,
        ),
        (
            axes[1, 1],
            df_gnn_x,
            "D. 10x Xenium: Mouse Kidney (Inter-Tubular Over-smoothing)",
            mlp_x_mean,
            mlp_x_std,
        ),
    ]

    for ax, df_curr, title, mlp_mean, mlp_std in panels:
        df_curr["Topology_Clean"] = df_curr["topology"].map(topo_map)
        df_curr["Model_Clean"] = df_curr["model"].str.upper()
        df_curr.loc[df_curr["Model_Clean"] == "GRAPHSAGE", "Model_Clean"] = "GraphSAGE"

        # Plot Parity Band
        parity_half = 2.0 * mlp_std
        ax.axhspan(
            mlp_mean - parity_half,
            mlp_mean + parity_half,
            color="#999999",
            alpha=0.25,
            label=f"MLP Baseline (±2σ: [{mlp_mean - parity_half:.3f}, {mlp_mean + parity_half:.3f}])",
            zorder=1,
        )
        ax.axhline(mlp_mean, color="#666666", linestyle="--", linewidth=1.5, zorder=2)

        # Plot grouped barplot with error bars
        sns.barplot(
            data=df_curr,
            x="Topology_Clean",
            y="test_macro_f1",
            hue="Model_Clean",
            order=topo_labels,
            palette=palette,
            ax=ax,
            capsize=0.08,
            err_kws={"linewidth": 1.2},
            zorder=3,
        )

        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
        ax.set_xlabel("Graph Construction / Topology", fontweight="bold", labelpad=8)
        ax.set_ylabel("Test Macro-F1 (10 Seeds)", fontweight="bold", labelpad=8)
        ax.set_xticks(range(len(topo_labels)))
        ax.set_xticklabels(topo_labels, rotation=35, ha="right", fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

        # Add vertical separator between canonical / bipartite / controls
        ax.axvline(1.5, color="#bbbbbb", linestyle=":", linewidth=1)
        ax.axvline(2.5, color="#bbbbbb", linestyle=":", linewidth=1)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()

    out_pdf = OUT_DIR / "fig1_cross_modality_benchmark.pdf"
    out_png = OUT_DIR / "fig1_cross_modality_benchmark.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1:\n  PDF: {out_pdf}\n  PNG: {out_png}")


if __name__ == "__main__":
    plot_cross_modality_matrix()
