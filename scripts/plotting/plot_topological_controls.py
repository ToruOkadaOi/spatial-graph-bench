"""Publication Figure 2: Topological Sensitivity and Negative Controls.

Demonstrates that GNNs actively compute on spatial topology rather than functioning
as randomized weight projections: coordinate scrambling and edge rewiring cause
catastrophic drops across isotropic architectures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def plot_topological_controls(
    dataset_name: str = "stereoseq_axolotl_telencephalon",
    split_id: str = "developmental_three_stage",
    out_suffix: str = "",
) -> None:
    csv_file = Path(f"results/reports/{dataset_name}_{split_id}_all_runs_summary.csv")
    if not csv_file.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {csv_file}")
    df = pd.read_csv(csv_file)
    df_sweep = df[df["run_type"] == "sweep"].copy()

    # Define control mappings
    control_types = {
        "spatial_knn_k6": "Canonical",
        "spatial_knn_k12": "Canonical",
        "shuffled_spatial_knn_k6": "Shuffled Coords",
        "shuffled_spatial_knn_k12": "Shuffled Coords",
        "rewired_spatial_knn_k6": "Rewired Edges",
        "rewired_spatial_knn_k12": "Rewired Edges",
    }
    k_vals = {
        "spatial_knn_k6": "k = 6",
        "spatial_knn_k12": "k = 12",
        "shuffled_spatial_knn_k6": "k = 6",
        "shuffled_spatial_knn_k12": "k = 12",
        "rewired_spatial_knn_k6": "k = 6",
        "rewired_spatial_knn_k12": "k = 12",
    }

    df_ctrl = df_sweep[df_sweep["topology"].isin(control_types.keys())].copy()
    df_ctrl["Condition"] = df_ctrl["topology"].map(control_types)
    df_ctrl["Neighborhood"] = df_ctrl["topology"].map(k_vals)
    df_ctrl["Model_Clean"] = df_ctrl["model"].str.upper()
    df_ctrl.loc[df_ctrl["Model_Clean"] == "GRAPHSAGE", "Model_Clean"] = "GraphSAGE"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    mlp_mean = df[df["model"] == "mlp"]["test_macro_f1"].mean()
    mlp_std = df[df["model"] == "mlp"]["test_macro_f1"].std()
    parity_half = 2.0 * mlp_std

    palette = {"Canonical": "#2ca02c", "Shuffled Coords": "#ff7f0e", "Rewired Edges": "#d62728"}

    for idx, (ax, k_str, panel_title) in enumerate(
        [
            (axes[0], "k = 6", "A. Topological Disruption (k = 6 Neighbors)"),
            (axes[1], "k = 12", "B. Topological Disruption (k = 12 Neighbors)"),
        ]
    ):
        df_k = df_ctrl[df_ctrl["Neighborhood"] == k_str]

        # Parity band
        ax.axhspan(
            mlp_mean - parity_half,
            mlp_mean + parity_half,
            color="#999999",
            alpha=0.25,
            label="MLP Baseline (±2σ)" if idx == 0 else None,
            zorder=1,
        )
        ax.axhline(mlp_mean, color="#666666", linestyle="--", linewidth=1.5, zorder=2)

        sns.barplot(
            data=df_k,
            x="Model_Clean",
            y="test_macro_f1",
            hue="Condition",
            palette=palette,
            ax=ax,
            capsize=0.08,
            err_kws={"linewidth": 1.2},
            zorder=3,
        )

        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("GNN Architecture", fontweight="bold", labelpad=8)
        if idx == 0:
            ax.set_ylabel("Test Macro-F1 (10 Seeds)", fontweight="bold", labelpad=8)
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
        ax.legend(
            title="Graph Topology", loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5
        )

    plt.tight_layout()

    out_pdf = OUT_DIR / f"fig2_topological_controls{out_suffix}.pdf"
    out_png = OUT_DIR / f"fig2_topological_controls{out_suffix}.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2:\n  PDF: {out_pdf}\n  PNG: {out_png}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot topological negative controls.")
    parser.add_argument("--dataset", default="stereoseq_axolotl_telencephalon")
    parser.add_argument("--split", default="developmental_three_stage")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    plot_topological_controls(
        dataset_name=args.dataset,
        split_id=args.split,
        out_suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
