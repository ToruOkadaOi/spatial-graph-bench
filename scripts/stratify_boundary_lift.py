"""Compatibility entry point for the full versioned prediction reanalysis."""

import argparse
import json

from spatial_graph_bench.analysis.boundary import compute_boundary_mask as compute_boundary_mask
from spatial_graph_bench.analysis.reanalysis import analysis_dir
from spatial_graph_bench.analysis.reanalysis import regenerate as reanalyze
from spatial_graph_bench.utils.paths import get_project_root


def run_post_hoc_analysis(
    dataset_name="merfish_mouse_spinal_cord",
    split_id="mouse_held_out_canonical",
    percentile=15.0,
    epsilon=None,
):
    if percentile != 15.0 or epsilon is not None:
        raise ValueError(
            "V2 uses a fixed 15% boundary and recomputed baseline margin across the full grid"
        )
    reanalyze()
    return json.loads(
        (
            analysis_dir(get_project_root(), dataset_name, split_id) / "post_hoc_summary.json"
        ).read_text()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="merfish_mouse_spinal_cord")
    parser.add_argument("--split", default="mouse_held_out_canonical")
    args = parser.parse_args()
    run_post_hoc_analysis(args.dataset, args.split)


if __name__ == "__main__":
    main()
