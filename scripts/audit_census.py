"""Audit dataset census manifest against local raw files and cryptographic anchors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_census(census_path: Path | str = "data/census.json") -> bool:
    census_file = Path(census_path)
    if not census_file.is_file():
        console.print(f"[bold red]Census file not found:[/bold red] {census_file}")
        return False

    with census_file.open("r", encoding="utf-8") as f:
        census = json.load(f)

    datasets = census.get("datasets", {})
    console.print(f"[bold cyan]Auditing Census Manifest: {census_file}[/bold cyan]")
    console.print(f"Total registered datasets: {len(datasets)}\n")

    table = Table(title="Dataset Census & Verification Gate Status")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Platform", style="magenta")
    table.add_column("Hierarchy Language", style="yellow")
    table.add_column("Local Raw File", style="white")
    table.add_column("SHA-256 Hash", style="blue")
    table.add_column("Gate Status", style="green")

    all_passed = True

    for ds_id, ds_info in datasets.items():
        platform = ds_info.get("platform", "unknown")
        hierarchy = ds_info.get("hierarchy_language", "unknown")
        files = ds_info.get("files", [])

        for file_entry in files:
            snapshot_id = file_entry.get("raw_data_snapshot_id")
            local_path = Path(snapshot_id) if snapshot_id else None
            expected_hash = file_entry.get("sha256")

            if local_path and local_path.is_file():
                actual_hash = compute_sha256(local_path)
                if expected_hash:
                    if actual_hash == expected_hash:
                        status = "[bold green]VERIFIED (MATCH)[/bold green]"
                    else:
                        status = "[bold red]CORRUPT (MISMATCH)[/bold red]"
                        all_passed = False
                else:
                    status = "[yellow]PRESENT (UNANCHORED)[/yellow]"
                hash_display = f"{actual_hash[:16]}..."
                file_display = (
                    f"{local_path.name} ({local_path.stat().st_size / (1024 * 1024):.1f} MB)"
                )
            else:
                status = "[dim yellow]REMOTE ANCHOR[/dim yellow]"
                hash_display = f"{expected_hash[:16]}..." if expected_hash else "n/a"
                file_display = file_entry.get("filename", "n/a")

            table.add_row(
                ds_id,
                platform,
                hierarchy,
                file_display,
                hash_display,
                status,
            )

    console.print(table)
    return all_passed


if __name__ == "__main__":
    census_arg = sys.argv[1] if len(sys.argv) > 1 else "data/census.json"
    ok = audit_census(census_arg)
    sys.exit(0 if ok else 1)
