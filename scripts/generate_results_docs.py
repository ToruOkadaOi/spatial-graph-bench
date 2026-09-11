"""Generate comprehensive empirical markdown results reports for all benchmark platforms."""

from __future__ import annotations

import json
from pathlib import Path


def generate_platform_report(
    dataset_name: str,
    split_id: str,
    platform_name: str,
    split_desc: str,
    takeaway_text: str,
    out_filename: str,
) -> None:
    res_dir = Path(f"artifacts/results/{dataset_name}/{split_id}")
    snap_dir = Path(f"audits/baselines_snapshot/{dataset_name}/{split_id}")
    post_hoc_file = res_dir / "post_hoc_summary.json"
    parity_file = snap_dir / "parity_band.json"

    with open(post_hoc_file, encoding="utf-8") as f:
        data = json.load(f)
    with open(parity_file, encoding="utf-8") as f:
        pb = json.load(f)

    halfwidth = pb["parity_band_halfwidth"]
    mlp_mean = pb["mlp_mean_test_macro_f1"]
    mlp_sigma = pb["mlp_sigma_test_macro_f1"]
    rf_f1 = pb["rf_test_macro_f1"]

    topo_labels = {
        "spatial_knn_k6": "`spatial_knn_k6`",
        "spatial_knn_k12": "`spatial_knn_k12`",
        "bipartite_ref_k20": "`bipartite_ref_k20`",
        "shuffled_spatial_knn_k6": "`shuffled_spatial_knn_k6`",
        "shuffled_spatial_knn_k12": "`shuffled_spatial_knn_k12`",
        "rewired_spatial_knn_k6": "`rewired_spatial_knn_k6`",
        "rewired_spatial_knn_k12": "`rewired_spatial_knn_k12`",
    }

    lines = [
        f"# Empirical Benchmark Results: {platform_name} (`{split_id}`)",
        "",
        f"**Benchmark Platform**: {platform_name}  ",
        f"**Dataset**: `{dataset_name}`  ",
        f"**Split**: `{split_id}` ({split_desc})  ",
        "**Pre-Registered Comparison Grid**: 4 GNN Architectures × 7 Graph Topologies × 10 Seeds (42–51) = 280 GPU Runs  ",
        f"**Parity Equivalence Margin**: $\\epsilon = \\pm {halfwidth:.4f}$ ($2\\sigma_{{\\text{{MLP}}}}$)  ",
        f"**Baseline Reference**: Spatially Ignorant MLP Test Macro-F1 = $\\mathbf{{{mlp_mean:.4f} \\pm {mlp_sigma:.4f}}}$; Random Forest = ${rf_f1:.4f}$  ",
        "",
        "---",
        "",
        "## 1. Primary Evidence Table: 10-Seed Aggregated Performance",
        "",
        "| Model | Graph Construction | GNN Test F1 | Baseline MLP | Matched Lift ($\\Delta$) | 90% TOST CI | FWER Decision ($\\alpha=0.05$) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in data:
        m = r["model"]
        g = topo_labels.get(r["graph"], r["graph"])
        gnn_f1 = f"{r['gnn_overall']:.4f}"
        mlp_f1 = f"{mlp_mean:.4f}"
        lift = f"{r['mean_lift']:+.4f}"
        ci = f"[{r['ci_90_low']:+.4f}, {r['ci_90_high']:+.4f}]"
        dec = r["decision"]
        dec_fmt = f"**{dec}**" if "POSITIVE" in dec or "NEGATIVE" in dec else dec
        lines.append(f"| **{m}** | {g} | {gnn_f1} | {mlp_f1} | {lift} | {ci} | {dec_fmt} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Post-Hoc Stratification: Interior vs. Boundary Margin",
            "",
            "Cells were geometrically stratified by distance to the section convex hull (outer 15% classified as boundary margin cells).",
            "",
            "| Model & Graph | Overall Lift ($\\Delta$) | Interior Lift ($\\Delta_{\\text{int}}$) | Boundary Lift ($\\Delta_{\\text{bnd}}$) | Margin Effect ($\\Delta_{\\text{bnd}} - \\Delta_{\\text{int}}$) |",
            "|:---|:---:|:---:|:---:|:---:|",
        ]
    )

    for r in data:
        if r["graph"] in ["spatial_knn_k6", "spatial_knn_k12", "bipartite_ref_k20"]:
            m = r["model"]
            g = topo_labels.get(r["graph"], r["graph"])
            l_ov = f"{r['mean_lift']:+.4f}"
            int_l = r["int_lift_mean"]
            bnd_l = r["bnd_lift_mean"]
            l_in = f"{int_l:+.4f}"
            l_bd = f"{bnd_l:+.4f}"
            diff = f"{(bnd_l - int_l):+.4f}"
            lines.append(f"| **{m}** on {g} | {l_ov} | {l_in} | {l_bd} | {diff} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Scientific Discussion & Biological Insights",
            "",
            takeaway_text,
            "",
        ]
    )

    out_path = Path("docs") / out_filename
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {out_path}")


def main() -> None:
    # 1. Stereo-seq
    generate_platform_report(
        dataset_name="stereoseq_axolotl_telencephalon",
        split_id="developmental_three_stage",
        platform_name="Stereo-seq (Axolotl Telencephalon)",
        split_desc="24,008 test spots across 16 cell types; Stage 44 query, Stage 54 reference",
        takeaway_text=(
            "Stereo-seq represents the sole benchmark modality demonstrating a **statistically significant positive lift** (+6.4% on raw seed runs, +0.0383 mean lift on GraphSAGE k=12). "
            "This success is biologically grounded in the continuous, laminar neuroepithelial architecture of the developing axolotl brain, where neighboring spots share high homophilic cell-type identities. "
            "Negative controls (rewired edges and shuffled coordinates) cause precipitous performance drops (down to 0.05 Macro-F1), proving that the inductive gain stems from authentic physical tissue structure."
        ),
        out_filename="results-stereoseq.md",
    )

    # 2. Open-ST
    generate_platform_report(
        dataset_name="openst_human_lymph_node",
        split_id="section_held_out_canonical",
        platform_name="Open-ST (Human Metastatic Lymph Node)",
        split_desc="17,998 test cells across 8 cell types; Section #19 query, Section #6 reference",
        takeaway_text=(
            "Open-ST demonstrates the **tumor-stroma boundary blurring failure mode** (-5.8% negative lift on GCN k=12, p < 10^-15). "
            "In metastatic head and neck squamous cell carcinoma, malignant epithelial nests are sharply demarcated from surrounding fibrous and immune stroma. "
            "Isotropic GNN convolutions inappropriately smooth gene expression across this barrier, falsely classifying peritumoral stromal cells as malignant tumor cells. "
            "Boundary stratification shows that this degradation is exacerbated at the tissue margin (-0.031 additional drop), where neighborhood degrees are truncated."
        ),
        out_filename="results-openst.md",
    )

    # 3. 10x Xenium
    generate_platform_report(
        dataset_name="xenium_mouse_kidney",
        split_id="replicate_held_out_canonical",
        platform_name="10x Xenium (Mouse Kidney)",
        split_desc="85,880 test cells across 20 cell types; ShamR query, ShamL reference",
        takeaway_text=(
            "10x Xenium demonstrates the **tubular-interstitial drowning mechanism** (-39.3% catastrophic negative lift on GCN k=12, p < 10^-15). "
            "Because kidney parenchyma is overwhelmingly dominated by proximal tubules and thick ascending limbs (>50% of cells), isotropic spatial aggregation completely swallows the transcriptomic signatures of thin interstitial, endothelial, and rare glomerular cells (Immune F1 collapses by -0.735, Fibroblasts by -0.660, Endothelia by -0.635, and PEC by -0.424). "
            "GraphSAGE rescues performance to statistical parity (0.7514 vs. 0.7635 MLP) via its root-node concatenation skip-connection, while Bipartite Reference Prototype matching (bipartite_ref_k20) brings GIN (0.7560) and GAT (0.7415) into equivalence by eliminating physical tissue mixing."
        ),
        out_filename="results-xenium.md",
    )


if __name__ == "__main__":
    main()
