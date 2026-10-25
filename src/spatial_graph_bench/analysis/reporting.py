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
    headline = []
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
        # Fixed k12 GraphSAGE illustration, not test-set selection of the best model.
        chosen = next(
            r for r in rows if r["model"] == "GRAPHSAGE" and r["graph"] == "spatial_knn_k12"
        )
        headline.append(
            f"| {platform} | {info['test_cells']:,} / {info['evaluated_cells']:,} | {len(info['evaluated_labels'])} | {info['label_coverage']:.2%} | {info['mlp_mean_test_macro_f1']:.4f} ± {info['mlp_sigma_test_macro_f1']:.4f} | {chosen['gnn_overall']:.4f} | {chosen['mean_lift']:+.4f} | {decision_label(chosen['decision'])} |"
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
    overview = [
        "# Spatial graph benchmark",
        "",
        "Does spatial message passing improve cell-type classification over expression-only baselines?",
        "",
        "**Status: historical results reanalyzed; strict-inductive GNN conclusions require corrected training.**",
        "",
        LIMITATION,
        "",
        "The preserved dataset contains 1,164 runs: four platforms × (280 GNN runs + ten MLP seeds + one random forest). Models use identical frozen PCA inputs within each split. These were fixed model configurations; a saved equal-budget tuning study is not available.",
        "",
        "The table illustrates the fixed GraphSAGE spatial k12 configuration, rather than selecting a winner on the test set. Statistical decisions describe the historical fits only.",
        "",
        "| Platform | Test / evaluated cells | Evaluated classes | Class coverage | MLP F1 ± sample SD | GraphSAGE k12 F1 | Paired lift | Holm decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *headline,
        "",
        "Stereo-seq holds out Stage57 after training on Stage44 and Stage54. MERFISH holds out mouse F4. Open-ST holds out section 19 after training on section 6. Xenium evaluates ShamL → ShamR; this split does not test injury-stage generalization.",
        "",
        "Test evaluation excludes true cells from unseen classes and averages over training/test intersection labels. Epsilon is max(0.005, twice the MLP sample SD). The reanalysis applies Holm adjustment to all 336 hypotheses across the four platforms. Equivalence and inconclusive evidence are separate outcomes. Boundary and per-class diagnostics are descriptive; they do not by themselves establish an oversmoothing mechanism.",
        "",
        "## Reproduce the analysis without training",
        "",
        "```bash",
        "uv sync --frozen --all-extras",
        "./scripts/reproduce_all.sh",
        "```",
        "",
        "The script fetches missing verified release bundles, validates the versioned byte index, recomputes all test metrics and statistics from predictions, then regenerates reports and figures. It never trains models or overwrites original run outputs. GitHub CLI access to the releases is needed when artifacts are absent. Runtime includes downloads and scales with network and disk speed; this is not a 30-second guarantee.",
        "",
        "For an existing offline artifact checkout: `./scripts/reproduce_all.sh --offline`. For reports only: `./scripts/reproduce_all.sh --offline --no-figures`.",
        "",
        "See [corrected analysis method](docs/reanalysis-v2.md), [all 112 comparisons](results/reports/table_benchmark_synthesis.md), [release index](docs/results-index.md), and [reproduction guide](REPRODUCE.md).",
        "",
        "Original predictions, probabilities, metrics, histories and manifests remain immutable in `artifacts/results/`. Corrected metrics and analysis inputs are versioned under `results/reports/reanalysis_v2/`. The new integrity index preserves and records legacy PCA hash discrepancies instead of rewriting old manifests.",
        "",
        "## Development",
        "",
        "```bash",
        "uv run --frozen pytest tests",
        "uv run --frozen ruff check src tests scripts",
        "uv run --frozen ruff format --check src tests scripts",
        "```",
        "",
        "MIT license. See [CITATION.md](CITATION.md) for citation information.",
    ]
    (root / "README.md").write_text("\n".join(overview) + "\n")
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
