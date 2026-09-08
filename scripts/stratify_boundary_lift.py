"""Post-Hoc Analysis: Boundary Stratification and Formal TOST Equivalence Testing (§3.1, §6.3).

Executes:
1. Geometric boundary margin identification on test tissue sections.
2. Interior vs. Boundary stratified Macro-F1 evaluation and matched lift.
3. Two One-Sided Tests (TOST) for equivalence against parity margin epsilon = 0.0069.
4. Superiority/Inferiority tests with Holm-Bonferroni FWER correction.
5. Emits structured JSON summary to artifacts/results/...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats
from rich.console import Console
from rich.table import Table
from scipy.spatial import ConvexHull
from sklearn.metrics import f1_score

console = Console()


def compute_boundary_mask(
    coords: np.ndarray, sections: np.ndarray, percentile: float = 15.0
) -> np.ndarray:
    """Classify cells into boundary-margin vs deep interior using distance to section hull."""

    def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=-1) / np.sum(ab * ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.linalg.norm(p - proj, axis=-1)

    is_boundary = np.zeros(len(coords), dtype=bool)
    for sec in np.unique(sections):
        sec_mask = sections == sec
        sec_pts = coords[sec_mask]
        hull = ConvexHull(sec_pts)

        min_dists = np.full(len(sec_pts), np.inf)
        for simplex in hull.simplices:
            a = sec_pts[simplex[0]]
            b = sec_pts[simplex[1]]
            d = point_to_segment_dist(sec_pts, a, b)
            min_dists = np.minimum(min_dists, d)

        threshold = np.percentile(min_dists, percentile)
        is_boundary[sec_mask] = min_dists <= threshold

    return is_boundary


def run_post_hoc_analysis(
    dataset_name: str = "merfish_mouse_spinal_cord",
    split_id: str = "mouse_held_out_canonical",
    percentile: float = 15.0,
    epsilon: float | None = None,
) -> list[dict[str, Any]]:
    results_dir = Path(f"artifacts/results/{dataset_name}/{split_id}")
    prep_dir = Path(f"artifacts/preprocessed/{dataset_name}/{split_id}")
    split_file = Path(f"splits/{dataset_name}/{split_id}.json")
    snapshot_dir = Path(f"audits/baselines_snapshot/{dataset_name}/{split_id}")

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed bundle directory not found: {prep_dir}")

    # Determine parity threshold epsilon
    if epsilon is None:
        parity_file = snapshot_dir / "parity_band.json"
        if parity_file.is_file():
            pdata = json.loads(parity_file.read_text(encoding="utf-8"))
            epsilon = float(pdata.get("parity_band_halfwidth", 0.0069))
        else:
            epsilon = 0.0069

    # Load preprocessed coordinates and labels
    coords = np.load(prep_dir / "spatial_test.npy")
    y_true = np.load(prep_dir / "test_labels.npy")

    with open(prep_dir / "cell_ids.json") as f:
        cell_info = json.load(f)
    sections = np.array(cell_info["sections_test"])

    with open(prep_dir / "label_mapping.json") as f:
        label_to_id = json.load(f)

    with open(split_file) as f:
        split_meta = json.load(f)

    eval_labels = split_meta["evaluated_labels"]
    eval_ids = [label_to_id[lab] for lab in eval_labels if lab in label_to_id]
    eval_mask = np.isin(y_true, eval_ids)

    # 1. Compute boundary mask
    is_boundary = compute_boundary_mask(coords, sections, percentile=percentile)
    int_mask = eval_mask & (~is_boundary)
    bnd_mask = eval_mask & is_boundary

    console.print(
        f"[bold green]>>> Loaded {len(coords)} test cells for {dataset_name} ({split_id}):[/bold green]"
    )
    console.print(
        f"  Interior evaluated cells: {int_mask.sum():,} ({int_mask.sum() / eval_mask.sum():.1%})"
    )
    console.print(
        f"  Boundary evaluated cells: {bnd_mask.sum():,} ({bnd_mask.sum() / eval_mask.sum():.1%})"
    )

    # 2. Evaluate MLP baselines (all available seeds)
    mlp_records = {}
    available_mlp_seeds = [
        seed
        for seed in range(42, 52)
        if (results_dir / f"mlp_none_seed{seed}" / "test_preds.npy").is_file()
        or (snapshot_dir / f"mlp_none_seed{seed}" / "test_preds.npy").is_file()
    ]
    if not available_mlp_seeds:
        raise FileNotFoundError(
            f"No MLP baseline test_preds.npy found under {results_dir} or {snapshot_dir}"
        )

    for seed in available_mlp_seeds:
        mlp_run_dir = results_dir / f"mlp_none_seed{seed}"
        if not (mlp_run_dir / "test_preds.npy").is_file():
            mlp_run_dir = snapshot_dir / f"mlp_none_seed{seed}"

        preds = np.load(mlp_run_dir / "test_preds.npy")

        overall_f1 = f1_score(
            y_true[eval_mask], preds[eval_mask], labels=eval_ids, average="macro", zero_division=0.0
        )
        int_f1 = f1_score(
            y_true[int_mask], preds[int_mask], labels=eval_ids, average="macro", zero_division=0.0
        )
        bnd_f1 = f1_score(
            y_true[bnd_mask], preds[bnd_mask], labels=eval_ids, average="macro", zero_division=0.0
        )

        mlp_records[seed] = {
            "overall_f1": float(overall_f1),
            "interior_f1": float(int_f1),
            "boundary_f1": float(bnd_f1),
        }

    # 3. Evaluate GNN configurations
    models = ["gcn", "gat", "gin", "graphsage"]
    graphs = [
        "spatial_knn_k6",
        "spatial_knn_k12",
        "rewired_spatial_knn_k6",
        "rewired_spatial_knn_k12",
        "shuffled_spatial_knn_k6",
        "shuffled_spatial_knn_k12",
        "bipartite_ref_k20",
    ]

    analysis_results: list[dict[str, Any]] = []

    for mod in models:
        for gr in graphs:
            overall_lifts = []
            int_lifts = []
            bnd_lifts = []
            gnn_overalls = []
            gnn_interiors = []
            gnn_boundaries = []

            for seed in available_mlp_seeds:
                run_dir = results_dir / f"{mod}_{gr}_seed{seed}"
                if not run_dir.exists():
                    continue
                preds = np.load(run_dir / "test_preds.npy")

                g_overall = f1_score(
                    y_true[eval_mask],
                    preds[eval_mask],
                    labels=eval_ids,
                    average="macro",
                    zero_division=0.0,
                )
                g_int = f1_score(
                    y_true[int_mask],
                    preds[int_mask],
                    labels=eval_ids,
                    average="macro",
                    zero_division=0.0,
                )
                g_bnd = f1_score(
                    y_true[bnd_mask],
                    preds[bnd_mask],
                    labels=eval_ids,
                    average="macro",
                    zero_division=0.0,
                )

                m_rec = mlp_records[seed]
                overall_lifts.append(g_overall - m_rec["overall_f1"])
                int_lifts.append(g_int - m_rec["interior_f1"])
                bnd_lifts.append(g_bnd - m_rec["boundary_f1"])

                gnn_overalls.append(g_overall)
                gnn_interiors.append(g_int)
                gnn_boundaries.append(g_bnd)

            n = len(overall_lifts)
            if n == 0:
                continue

            arr_lift = np.array(overall_lifts)
            mean_lift = float(np.mean(arr_lift))
            std_lift = float(np.std(arr_lift, ddof=1)) if n > 1 else 0.0
            se = std_lift / np.sqrt(n) if n > 1 and std_lift > 0 else 1e-6

            if n > 1 and se > 1e-6:
                # TOST tests: H01: Delta <= -eps, H02: Delta >= +eps
                t1 = (mean_lift - (-epsilon)) / se
                t2 = (mean_lift - epsilon) / se
                p1 = 1.0 - stats.t.cdf(t1, df=n - 1)
                p2 = stats.t.cdf(t2, df=n - 1)
                p_tost = float(max(p1, p2))

                # Directional hypothesis tests
                # Superiority: H0: Delta <= eps
                p_sup = float(1.0 - stats.t.cdf(t2, df=n - 1))
                # Inferiority: H0: Delta >= -eps
                p_inf = float(stats.t.cdf(t1, df=n - 1))

                # 90% Confidence Interval for TOST
                t_crit_90 = stats.t.ppf(0.95, df=n - 1)
                ci_90_low = mean_lift - t_crit_90 * se
                ci_90_high = mean_lift + t_crit_90 * se
            else:
                p_tost = 1.0
                p_sup = 1.0
                p_inf = 1.0
                ci_90_low = mean_lift
                ci_90_high = mean_lift

            int_arr = np.array(int_lifts)
            bnd_arr = np.array(bnd_lifts)

            analysis_results.append(
                {
                    "model": mod.upper(),
                    "graph": gr,
                    "n": n,
                    "gnn_overall": float(np.mean(gnn_overalls)),
                    "gnn_int": float(np.mean(gnn_interiors)),
                    "gnn_bnd": float(np.mean(gnn_boundaries)),
                    "mean_lift": mean_lift,
                    "std_lift": std_lift,
                    "ci_90_low": float(ci_90_low),
                    "ci_90_high": float(ci_90_high),
                    "int_lift_mean": float(np.mean(int_arr)),
                    "bnd_lift_mean": float(np.mean(bnd_arr)),
                    "p_tost": p_tost,
                    "p_sup": p_sup,
                    "p_inf": p_inf,
                }
            )

    if not analysis_results:
        console.print("[yellow]No GNN runs found to analyze.[/yellow]")
        return []

    # 4. Apply Holm-Bonferroni FWER correction across all evaluated configurations
    m_tests = len(analysis_results)
    indexed_results = []
    for idx, r in enumerate(analysis_results):
        min_p = min(r["p_tost"], r["p_sup"], r["p_inf"])
        if min_p == r["p_sup"]:
            t_type = "superiority"
        elif min_p == r["p_inf"]:
            t_type = "inferiority"
        else:
            t_type = "equivalence"
        indexed_results.append((min_p, t_type, idx))

    indexed_results.sort(key=lambda x: x[0])

    for rank, (p_raw, t_type, orig_idx) in enumerate(indexed_results):
        r = analysis_results[orig_idx]
        alpha_hb = 0.05 / (m_tests - rank)
        sig = p_raw < alpha_hb
        r["fwer_alpha"] = float(alpha_hb)
        r["fwer_significant"] = bool(sig)
        if t_type == "superiority" and sig:
            r["decision"] = "POSITIVE LIFT"
        elif t_type == "inferiority" and sig:
            r["decision"] = "NEGATIVE LIFT"
        elif t_type == "equivalence" and sig:
            r["decision"] = "PARITY (EQUIVALENT)"
        else:
            r["decision"] = "INCONCLUSIVE / PARITY"

    # Display Table 1: Stratified Lift (Interior vs. Boundary)
    t1 = Table(
        title=f"Post-Hoc Boundary Stratification: Interior vs. Boundary Lift ({dataset_name} / {split_id})"
    )
    t1.add_column("Model", style="cyan")
    t1.add_column("Graph Construction", style="magenta")
    t1.add_column("Overall Lift (Δ)", justify="right")
    t1.add_column("Interior Lift (Δ_int)", justify="right")
    t1.add_column("Boundary Lift (Δ_bnd)", justify="right")
    t1.add_column("Margin Effect (Δ_bnd - Δ_int)", justify="right", style="bold")

    for r in analysis_results:
        margin_diff = r["bnd_lift_mean"] - r["int_lift_mean"]
        diff_color = "green" if margin_diff > 0 else "red"
        t1.add_row(
            str(r["model"]),
            str(r["graph"]),
            f"{r['mean_lift']:+.4f} ± {r['std_lift']:.4f}",
            f"{r['int_lift_mean']:+.4f}",
            f"{r['bnd_lift_mean']:+.4f}",
            f"[{diff_color}]{margin_diff:+.4f}[/{diff_color}]",
        )
    console.print(t1)

    # Display Table 2: Formal TOST & Holm-Bonferroni Hypothesis Decisions
    t2 = Table(
        title=f"Formal TOST & Holm-Bonferroni Hypothesis Decisions (Parity Margin ±{epsilon:.4f})"
    )
    t2.add_column("Model", style="cyan")
    t2.add_column("Graph Construction", style="magenta")
    t2.add_column("90% TOST CI", justify="center")
    t2.add_column("Raw p-value", justify="right")
    t2.add_column("FWER Decision (α=0.05)", justify="center", style="bold")

    for r in analysis_results:
        ci_str = f"[{r['ci_90_low']:+.4f}, {r['ci_90_high']:+.4f}]"
        if "POSITIVE" in r["decision"]:
            dec_color = "green"
            p_val_str = f"p_sup={r['p_sup']:.2e}"
        elif "NEGATIVE" in r["decision"]:
            dec_color = "red"
            p_val_str = f"p_inf={r['p_inf']:.2e}"
        else:
            dec_color = "yellow"
            p_val_str = f"p_tost={r['p_tost']:.2e}"

        t2.add_row(
            str(r["model"]),
            str(r["graph"]),
            ci_str,
            p_val_str,
            f"[{dec_color}]{r['decision']}[/{dec_color}]",
        )
    console.print(t2)

    # Save output
    out_file = results_dir / "post_hoc_summary.json"
    with open(out_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    console.print(f"\n[bold green]✓ Post-hoc analysis saved to:[/bold green] {out_file}")
    return analysis_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", type=str, default="mouse_held_out_canonical")
    parser.add_argument("--percentile", type=float, default=15.0)
    parser.add_argument("--epsilon", type=float, default=None)
    args = parser.parse_args()

    run_post_hoc_analysis(
        dataset_name=args.dataset,
        split_id=args.split,
        percentile=args.percentile,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
