"""Inspect corrected historical results, keeping inconclusive separate from equivalence."""

import argparse
import json
from pathlib import Path

from spatial_graph_bench.analysis.reanalysis import analysis_dir
from spatial_graph_bench.utils.paths import get_project_root

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/results/merfish_mouse_spinal_cord/mouse_held_out_canonical"),
    )
    a = p.parse_args()
    ds, sp = a.results_dir.parts[-2:]
    rows = json.loads(
        (analysis_dir(get_project_root(), ds, sp) / "post_hoc_summary.json").read_text()
    )
    for row in rows:
        print(
            f"{row['model']:10} {row['graph']:30} F1={row['gnn_overall']:.4f} lift={row['mean_lift']:+.4f} {row['decision']}"
        )
