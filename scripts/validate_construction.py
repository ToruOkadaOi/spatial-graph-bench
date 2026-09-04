"""Validate graph construction bundles and assert protocol connectivity invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spatial_graph_bench.graph.audit import validate_graph_construction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path)
    args = parser.parse_args()

    ok = validate_graph_construction(args.graph_dir)
    sys.exit(0 if ok else 1)
