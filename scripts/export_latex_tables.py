"""Export publication-ready LaTeX tables for cross-modality benchmark synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_latex_table() -> None:
    merfish_ph = Path(
        "artifacts/results/merfish_mouse_spinal_cord/mouse_held_out_canonical/post_hoc_summary.json"
    )
    stereoseq_ph = Path(
        "artifacts/results/stereoseq_axolotl_telencephalon/developmental_three_stage/post_hoc_summary.json"
    )
    openst_ph = Path(
        "artifacts/results/openst_human_lymph_node/section_held_out_canonical/post_hoc_summary.json"
    )
    xenium_ph = Path(
        "artifacts/results/xenium_mouse_kidney/replicate_held_out_canonical/post_hoc_summary.json"
    )

    with open(merfish_ph) as f:
        m_data = json.load(f)
    with open(stereoseq_ph) as f:
        s_data = json.load(f)
    with open(openst_ph) as f:
        o_data = json.load(f)
    with open(xenium_ph) as f:
        x_data = json.load(f)

    topo_order = [
        "spatial_knn_k6",
        "spatial_knn_k12",
        "bipartite_ref_k20",
        "shuffled_spatial_knn_k6",
        "shuffled_spatial_knn_k12",
        "rewired_spatial_knn_k6",
        "rewired_spatial_knn_k12",
    ]
    topo_labels = {
        "spatial_knn_k6": "Spatial $k=6$",
        "spatial_knn_k12": "Spatial $k=12$",
        "bipartite_ref_k20": "Bipartite Ref $k=20$",
        "shuffled_spatial_knn_k6": "Shuffled $k=6$",
        "shuffled_spatial_knn_k12": "Shuffled $k=12$",
        "rewired_spatial_knn_k6": "Rewired $k=6$",
        "rewired_spatial_knn_k12": "Rewired $k=12$",
    }
    model_order = ["GCN", "GAT", "GIN", "GRAPHSAGE"]
    model_labels = {"GCN": "GCN", "GAT": "GAT", "GIN": "GIN", "GRAPHSAGE": "GraphSAGE"}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Quad-Modality Benchmark Synthesis: Statistical Equivalence and Matched Lift.} Evaluation of 4 GNN architectures across 7 topologies ($N=10$ random seeds) against frozen non-spatial MLP baselines across four spatial transcriptomics technologies. Two One-Sided Tests (TOST) evaluate equivalence within pre-registered parity bands ($\epsilon_{\text{MERFISH}} = \pm 0.0069$, $\epsilon_{\text{Stereo-seq}} = \pm 0.0215$, $\epsilon_{\text{Open-ST}} = \pm 0.0054$, $\epsilon_{\text{Xenium}} = \pm 0.0241$). Directional significance is reported with Holm-Bonferroni FWER control ($\alpha = 0.05$).}",
        r"\label{tab:benchmark_synthesis}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Modality / Dataset} & \textbf{Model} & \textbf{Topology} & \textbf{GNN Macro-F1} & \textbf{Matched Lift $\Delta$} & \textbf{90\% TOST CI} & \textbf{$p_{\text{TOST}}$} & \textbf{FWER Decision} \\",
        r"\midrule",
    ]

    for ds_name, data in [
        ("Stereo-seq (Axolotl Telencephalon)", s_data),
        ("MERFISH (Mouse Spinal Cord)", m_data),
        ("Open-ST (Human Metastatic Lymph Node)", o_data),
        ("10x Xenium (Mouse Kidney)", x_data),
    ]:
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{ds_name}}}}} \\\\")
        lines.append(r"\midrule")
        df = pd.DataFrame(data)

        for m in model_order:
            for topo in topo_order:
                sub = df[(df["model"] == m) & (df["graph"] == topo)]
                if sub.empty:
                    continue
                row = sub.iloc[0]

                gnn_f1_str = f"{row['gnn_overall']:.4f}"
                lift_val = row["mean_lift"]
                lift_str = f"{lift_val:+.4f}"
                ci_str = f"[{row['ci_90_low']:+.4f}, {row['ci_90_high']:+.4f}]"
                p_tost_str = f"{row['p_tost']:.4f}" if row["p_tost"] >= 0.0001 else r"$<10^{-4}$"
                decision = row["decision"]

                # Formatting decision
                if "POSITIVE" in decision:
                    dec_str = r"\textbf{Positive Lift}$^*$"
                elif "PARITY" in decision or "EQUIVALENT" in decision:
                    dec_str = r"Parity"
                else:
                    dec_str = r"Negative Lift"

                m_label = model_labels[m]
                t_label = topo_labels[topo]
                lines.append(
                    f" & {m_label} & {t_label} & {gnn_f1_str} & {lift_str} & {ci_str} & {p_tost_str} & {dec_str} \\\\"
                )
            lines.append(r"\addlinespace[0.3em]")
        lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    out_file = Path("results/reports/table_benchmark_synthesis.tex")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated LaTeX table:\n  {out_file}")

    # Generate Markdown version
    md_lines = [
        "# Quad-Modality Benchmark Synthesis Table",
        "",
        "Evaluation of 4 GNN architectures across 7 topologies ($N=10$ random seeds) against frozen non-spatial MLP baselines across four spatial transcriptomics technologies.",
        "",
        "| Modality / Dataset | Model | Topology | GNN Macro-F1 | Matched Lift $\\Delta$ | 90% TOST CI | $p_{\\text{TOST}}$ | FWER Decision |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---|",
    ]

    for ds_name, data in [
        ("Stereo-seq (Axolotl Telencephalon)", s_data),
        ("MERFISH (Mouse Spinal Cord)", m_data),
        ("Open-ST (Human Metastatic Lymph Node)", o_data),
        ("10x Xenium (Mouse Kidney)", x_data),
    ]:
        md_lines.append(f"| **{ds_name}** | | | | | | | |")
        df = pd.DataFrame(data)
        for m in model_order:
            for topo in topo_order:
                sub = df[(df["model"] == m) & (df["graph"] == topo)]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                gnn_f1_str = f"{row['gnn_overall']:.4f}"
                lift_str = f"{row['mean_lift']:+.4f}"
                ci_str = f"[{row['ci_90_low']:+.4f}, {row['ci_90_high']:+.4f}]"
                p_tost_str = f"{row['p_tost']:.4f}" if row["p_tost"] >= 0.0001 else "<1e-4"
                decision = row["decision"]

                if "POSITIVE" in decision:
                    dec_str = "**Positive Lift**"
                elif "PARITY" in decision or "EQUIVALENT" in decision:
                    dec_str = "Parity"
                else:
                    dec_str = "Negative Lift"

                m_label = model_labels[m]
                t_label = topo_labels[topo]
                md_lines.append(
                    f"| | {m_label} | {t_label} | {gnn_f1_str} | {lift_str} | {ci_str} | {p_tost_str} | {dec_str} |"
                )

    out_md = Path("results/reports/table_benchmark_synthesis.md")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Generated Markdown table:\n  {out_md}")


if __name__ == "__main__":
    generate_latex_table()
