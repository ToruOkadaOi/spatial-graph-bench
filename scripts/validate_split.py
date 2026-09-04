"""Validate frozen split definition files for disjointness and §3.4 label-space coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from spatial_graph_bench.splitting.schema import SplitDefinition

console = Console()


def validate_split_file(split_path: Path) -> bool:
    console.print(f"[bold cyan]Validating Split File:[/bold cyan] {split_path}")
    if not split_path.is_file():
        console.print(f"[bold red]Split file missing:[/bold red] {split_path}")
        return False

    try:
        split = SplitDefinition.load_json(split_path)
        split.validate_disjointness()

        table = Table(title="Split Integrity Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        table.add_row("Dataset Name", split.dataset_name, "OK")
        table.add_row("Split ID", split.split_id, "OK")
        table.add_row("Hierarchy", split.hierarchy.value, "OK")
        table.add_row(
            "Cell Counts",
            f"Train: {len(split.train_cell_ids)}, Val: {len(split.val_cell_ids)}, Test: {len(split.test_cell_ids)}",
            "DISJOINT",
        )
        table.add_row(
            "Label Space (§3.4)",
            f"Evaluated: {len(split.evaluated_labels)}, Excluded: {len(split.excluded_labels)} (Coverage: {split.coverage_ratio:.1%})",
            "OK",
        )
        table.add_row("Artifact Hash", f"{split.compute_artifact_hash()[:16]}...", "VALID")

        console.print(table)
        return True
    except Exception as err:
        console.print(f"[bold red]Validation failed with error:[/bold red] {err}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_file", type=Path)
    args = parser.parse_args()

    ok = validate_split_file(args.split_file)
    sys.exit(0 if ok else 1)
