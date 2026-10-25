"""Generate all primary narrative and tables from versioned prediction reanalysis."""

import json
from pathlib import Path

import numpy as np

from spatial_graph_bench.analysis.catalog import DATASETS, LIMITATION, SEEDS
from spatial_graph_bench.analysis.reanalysis import analysis_dir


def decision_label(decision: str) -> str:
    labels = {
        "POSITIVE LIFT": "Positive lift",
        "NEGATIVE LIFT": "Negative lift",
        "EQUIVALENT": "Equivalent",
        "INCONCLUSIVE": "Inconclusive",
    }
    return labels[decision]


def generate_reports(root: Path) -> None:
    table = [
        "# Historical prediction reanalysis (v2)",
        "",
        LIMITATION,
        "",
        "All reported test scores use the train/test label intersection; unseen true cells are excluded uniformly.",
        "Paired seeds 42–51; sample SD. Unadjusted 90% CIs are descriptive. Holm adjusted p-values cover all 336 hypotheses (112 configurations × superiority, inferiority, equivalence). This is a retrospective analysis amendment.",
        "",
        "| Dataset | Model | Topology | GNN F1 | MLP F1 | Matched lift | 90% CI | Adjusted p (sup / inf / equiv) | Decision |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    latex = [
        r"\begin{longtable}{lllrrrrl}",
        r"Dataset & Model & Topology & GNN F1 & MLP F1 & Lift & $p_{\mathrm{Holm}}$ & Decision \\",
    ]
    dataset_reports = []
    for ds, sp, platform in DATASETS:
        output = analysis_dir(root, ds, sp)
        info = json.loads((output / "baseline_summary.json").read_text())
        rows = json.loads((output / "post_hoc_summary.json").read_text())
        scores = json.loads((output / "test_metrics.json").read_text())
        lines = [
            f"# {platform}: historical prediction reanalysis",
            "",
            LIMITATION,
            "",
            f"Dataset: `{ds}`; split: `{sp}`.",
            f"Training groups: {', '.join(info['train_groups'])}; held-out test groups: {', '.join(info['test_groups'])}.",
            f"Test cells/spots: {info['test_cells']:,}; evaluated: {info['evaluated_cells']:,}; intersection classes: {len(info['evaluated_labels'])}; class coverage: {info['label_coverage']:.2%}.",
            f"Excluded test classes: {', '.join(info['excluded_labels']) or 'none'}.",
            f"MLP test F1: {info['mlp_mean_test_macro_f1']:.4f} ± {info['mlp_sigma_test_macro_f1']:.4f}; RF: {info['rf_test_macro_f1']:.4f}; epsilon = max(0.005, 2 × sample SD) = {info['tost_equivalence_margin_epsilon']:.4f}.",
            "",
            "Boundary diagnostics use the outer 15% by distance to each section's convex hull; this is not an internal histological boundary annotation.",
            "",
            "| Model | Topology | GNN F1 | Paired lift | 90% CI (unadjusted) | Holm decision | Interior lift | Boundary lift |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
        ]
        for r in rows:
            ci = f"[{r['ci_90_low']:+.4f}, {r['ci_90_high']:+.4f}]"
            decision = decision_label(r["decision"])
            table.append(
                f"| {platform} | {r['model']} | {r['graph']} | {r['gnn_overall']:.4f} | {r['mlp_overall']:.4f} | {r['mean_lift']:+.4f} | {ci} | {r['p_sup_holm']:.3g} / {r['p_inf_holm']:.3g} / {r['p_tost_holm']:.3g} | {decision} |"
            )
            lines.append(
                f"| {r['model']} | {r['graph']} | {r['gnn_overall']:.4f} | {r['mean_lift']:+.4f} | {ci} | {decision} | {r['int_lift_mean']:+.4f} | {r['bnd_lift_mean']:+.4f} |"
            )
            p_key = {
                "POSITIVE LIFT": "p_sup_holm",
                "NEGATIVE LIFT": "p_inf_holm",
                "EQUIVALENT": "p_tost_holm",
            }.get(r["decision"])
            p = f"{r[p_key]:.3g}" if p_key else "--"
            topology = r["graph"].replace("_", r"\_")
            latex.append(
                f"{platform} & {r['model']} & {topology} & {r['gnn_overall']:.4f} & {r['mlp_overall']:.4f} & {r['mean_lift']:+.4f} & {p} & {decision} "
                + r"\\"
            )
            # Descriptive per-class results, using the same filtered predictions.
            model, graph = r["model"].lower(), r["graph"]
            classes = []
            for name in info["evaluated_labels"]:
                g = np.array(
                    [scores[f"{model}_{graph}_seed{s}"]["per_class_f1"][name] for s in SEEDS]
                )
                m = np.array([scores[f"mlp_none_seed{s}"]["per_class_f1"][name] for s in SEEDS])
                classes.append(
                    {
                        "class_name": name,
                        "support": info["test_label_support"][name],
                        "gnn_f1_mean": float(g.mean()),
                        "gnn_f1_std": float(g.std(ddof=1)),
                        "mlp_f1_mean": float(m.mean()),
                        "mlp_f1_std": float(m.std(ddof=1)),
                        "lift_mean": float((g - m).mean()),
                        "lift_std": float((g - m).std(ddof=1)),
                    }
                )
            (output / f"per_class_lift_{model}_{graph}.json").write_text(
                json.dumps({"classes": classes, "overall_lift": r["mean_lift"]}, indent=2) + "\n"
            )
        doc = (
            root
            / "docs"
            / f"results-{platform.lower().replace('stereo-seq', 'stereoseq').replace('open-st', 'openst')}.md"
        )
        doc.write_text("\n".join(lines) + "\n")
        dataset_reports.append(f"- [{platform}]({doc.name})")
    latex.append(r"\end{longtable}")
    (root / "results/reports/table_benchmark_synthesis.md").write_text("\n".join(table) + "\n")
    (root / "results/reports/table_benchmark_synthesis.tex").write_text(
        "% Historical fits; see docs/reanalysis-v2.md for limitations and method.\n"
        + "\n".join(latex)
        + "\n"
    )
    # README is a hand-edited project overview; analysis regeneration leaves it intact.
    (root / "docs/results-synthesis.md").write_text(
        "# Historical benchmark synthesis\n\n"
        + LIMITATION
        + "\n\n"
        + "See the [v2 method](reanalysis-v2.md) and [complete comparison table](../results/reports/table_benchmark_synthesis.md).\n\n"
        + "\n".join(dataset_reports)
        + "\n"
    )
    assets = json.loads((root / "configs/release_assets.json").read_text())
    release_lines = [
        "# Release artifacts and corrected analysis",
        "",
        LIMITATION,
        "",
        "MERFISH, Stereo-seq and Open-ST v1 result archives omit baselines; Xenium includes them. Four local Stereo-seq runs also differ from the old release. The exact missing/differing local historical metrics, manifests and predictions are preserved in `audits/reanalysis_recovery_v2.zip`, with reasons listed in its JSON manifest and bytes verified against the integrity index. Probabilities and histories are optional for prediction reanalysis; any present originals are still checked.",
        "",
        "| Dataset | Kind | Release | Archive SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for asset in assets:
        tag = asset["tag"]
        release_lines.append(
            f"| {asset['dataset']} | {asset['kind']} | [{tag}](https://github.com/ToruOkadaOi/spatial-graph-bench/releases/tag/{tag}) | `{asset['sha256']}` |"
        )
    release_lines += [
        "",
        "Corrected scores and sample counts are generated from frozen arrays in the [README](../README.md) and [dataset reports](results-synthesis.md). Original baseline summaries remain historical records of their original metric convention.",
    ]
    (root / "docs/results-index.md").write_text("\n".join(release_lines) + "\n")
