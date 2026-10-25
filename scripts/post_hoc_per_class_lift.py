"""Inspect descriptive per-class metrics from the consistent v2 label policy."""

import argparse
import json

from spatial_graph_bench.analysis.reanalysis import analysis_dir
from spatial_graph_bench.utils.paths import get_project_root


def analyze_per_class_lift(
    dataset_name="stereoseq_axolotl_telencephalon",
    split_id="developmental_three_stage",
    model="graphsage",
    graph="spatial_knn_k12",
    seeds=None,
):
    if seeds is not None and list(seeds) != list(range(42, 52)):
        raise ValueError("V2 uses the complete frozen ten-seed set")
    path = (
        analysis_dir(get_project_root(), dataset_name, split_id)
        / f"per_class_lift_{model}_{graph}.json"
    )
    return json.loads(path.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="stereoseq_axolotl_telencephalon")
    p.add_argument("--split", default="developmental_three_stage")
    p.add_argument("--model", default="graphsage")
    p.add_argument("--graph", default="spatial_knn_k12")
    a = p.parse_args()
    print(json.dumps(analyze_per_class_lift(a.dataset, a.split, a.model, a.graph), indent=2))
