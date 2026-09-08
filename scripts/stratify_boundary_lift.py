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
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats
from rich.console import Console
from rich.table import Table
from scipy.spatial import ConvexHull
from sklearn.metrics import f1_score

console = Console()


def compute_boundary_mask(coords: np.ndarray, sections: np.ndarray, percentile: float = 15.0) -> np.ndarray:
    """Classify cells into boundary-margin vs deep interior using distance to section hull."""
    def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=-1) / np.sum(ab * ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.linalg.norm(p - proj, axis=-1)

    is_boundary = np.zeros(len(coords), dtype=bool)
    for sec in np.unique(sections):
        sec_mask = (sections == sec)
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


def run_post_hoc_analysis():
    results_dir = Path("artifacts/results/merfish_mouse_spinal_cord/mouse_held_out_canonical")
    prep_dir = Path("artifacts/preprocessed/merfish_mouse_spinal_cord/mouse_held_out_canonical")
    split_file = Path("splits/merfish_mouse_spinal_cord/mouse_held_out_canonical.json")

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
    is_boundary = compute_boundary_mask(coords, sections, percentile=15.0)
    int_mask = eval_mask & (~is_boundary)
    bnd_mask = eval_mask & is_boundary

    console.print(f"[bold green]>>> Loaded {len(coords)} test cells:[/bold green]")
    console.print(f"  Interior evaluated cells: {int_mask.sum():,} ({int_mask.sum()/eval_mask.sum():.1%})")
    console.print(f"  Boundary evaluated cells: {bnd_mask.sum():,} ({bnd_mask.sum()/eval_mask.sum():.1%})")

    # 2. Evaluate MLP baselines (10 seeds)
    mlp_records = {}
    for seed in range(42, 52):
        mlp_run_dir = results_dir / f"mlp_none_seed{seed}"
        preds = np.load(mlp_run_dir / "test_preds.npy")

        overall_f1 = f1_score(y_true[eval_mask], preds[eval_mask], labels=eval_ids, average="macro", zero_division=0.0)
        int_f1 = f1_score(y_true[int_mask], preds[int_mask], labels=eval_ids, average="macro", zero_division=0.0)
        bnd_f1 = f1_score(y_true[bnd_mask], preds[bnd_mask], labels=eval_ids, average="macro", zero_division=0.0)

        mlp_records[seed] = {
            "overall_f1": float(overall_f1),
            "interior_f1": float(int_f1),
            "boundary_f1": float(bnd_f1),
        }

    # 3. Evaluate 28 GNN configurations (10 seeds each)
    models = ["gcn", "gat", "gin", "graphsage"]
    graphs = [
        "spatial_knn_k6", "spatial_knn_k12",
        "rewired_spatial_knn_k6", "rewired_spatial_knn_k12",
        "shuffled_spatial_knn_k6", "shuffled_spatial_knn_k12",
        "bipartite_ref_k20",
    ]

    analysis_results = []
    epsilon = 0.0069  # Pre-registered parity threshold

    for mod in models:
        for gr in graphs:
            overall_lifts = []
            int_lifts = []
            bnd_lifts = []
            gnn_overalls = []
            gnn_interiors = []
            gnn_boundaries = []

            for seed in range(42, 52):
                run_dir = results_dir / f"{mod}_{gr}_seed{seed}"
                if not run_dir.exists():
                    continue
                preds = np.load(run_dir / "test_preds.npy")

                g_overall = f1_score(y_true[eval_mask], preds[eval_mask], labels=eval_ids, average="macro", zero_division=0.0)
                g_int = f1_score(y_true[int_mask], preds[int_mask], labels=eval_ids, average="macro", zero_division=0.0)
                g_bnd = f1_score(y_true[bnd_mask], preds[bnd_mask], labels=eval_ids, average="macro", zero_division=0.0)

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
            std_lift = float(np.std(arr_lift, ddof=1))
            se = std_lift / np.sqrt(n)

            # TOST tests: H01: Delta <= -eps, H02: Delta >= +eps
            t1 = (mean_lift - (-epsilon)) / se
            t2 = (mean_lift - epsilon) / se
            p1 = 1.0 - stats.t.cdf(t1, df=n-1)
            p2 = stats.t.cdf(t2, df=n-1)
            p_tost = float(max(p1, p2))

            # Directional hypothesis tests
            # Superiority: H0: Delta <= eps
            p_sup = float(1.0 - stats.t.cdf(t2, df=n-1))
            # Inferiority: H0: Delta >= -eps
            p_inf = float(stats.t.cdf(t1, df=n-1))

            # 90% Confidence Interval for TOST
            t_crit_90 = stats.t.ppf(0.95, df=n-1)
            ci_90_low = mean_lift - t_crit_90 * se
            ci_90_high = mean_lift + t_crit_90 * se

            int_arr = np.array(int_lifts)
            bnd_arr = np.array(bnd_lifts)

            analysis_results.append({
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
                "int_lift_std": float(np.std(int_arr, ddof=1)),
                "bnd_lift_mean": float(np.mean(bnd_arr)),
                "bnd_lift_std": float(np.std(bnd_arr, ddof=1)),
                "p_tost": p_tost,
                "p_sup": p_sup,
                "p_inf": p_inf,
            })

    # 4. Holm-Bonferroni correction over the 28 comparisons
    # For inferiority (if negative) or superiority (if positive)
    m_tests = len(analysis_results)
    # Collect unadjusted p-values for testing hypothesis against parity band
    p_to_correct = []
    for r in analysis_results:
        if r["mean_lift"] > epsilon:
            p_val = r["p_sup"]
            test_type = "superiority"
        elif r["mean_lift"] < -epsilon:
            p_val = r["p_inf"]
            test_type = "inferiority"
        else:
            p_val = r["p_tost"]
            test_type = "equivalence"
        p_to_correct.append((p_val, test_type, r))

    # Sort ascending
    p_to_correct.sort(key=lambda x: x[0])
    for rank, (p_raw, t_type, r) in enumerate(p_to_correct):
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
    t1 = Table(title="Post-Hoc Boundary Stratification: Interior vs. Boundary Lift (merfish_canonical)")
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
            r["model"],
            r["graph"],
            f"{r['mean_lift']:+.4f} ± {r['std_lift']:.4f}",
            f"{r['int_lift_mean']:+.4f}",
            f"{r['bnd_lift_mean']:+.4f}",
            f"[{diff_color}]{margin_diff:+.4f}[/{diff_color}]",
        )
    console.print(t1)

    # Display Table 2: Formal TOST & Holm-Bonferroni Hypothesis Decisions
    t2 = Table(title="Formal TOST & Holm-Bonferroni Hypothesis Decisions (Parity Margin ±0.0069)")
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
            r["model"],
            r["graph"],
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


if __name__ == "__main__":
    run_post_hoc_analysis()
