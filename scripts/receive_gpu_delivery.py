"""Receive, verify through 4 layers, and ingest a packaged GPU results delivery."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from spatial_graph_bench.analysis.delivery import (
    AuditVerdict,
    BatchManifest,
    RunAuditReport,
    append_ingestion_log,
    audit_run_dir,
    verify_batch_files,
)
from spatial_graph_bench.preprocessing.schema import PreprocessedBundle
from spatial_graph_bench.utils.paths import ArtifactPaths

console = Console()


def receive_delivery(
    source: Path,
    dataset_name: str | None = None,
    split_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    paths = ArtifactPaths.default()
    staging_root = Path(tempfile.mkdtemp(prefix="spatial_recv_"))

    try:
        if source.is_file() and source.suffix == ".gz":
            console.print(f"[cyan]Extracting {source.name}...[/cyan]")
            with tarfile.open(source, "r:gz") as tar:
                tar.extractall(staging_root, filter="data")
            batch_dir = (
                staging_root
                if (staging_root / "batch_manifest.json").is_file()
                else list(staging_root.iterdir())[0]
            )
        elif source.is_dir():
            batch_dir = source
        else:
            raise FileNotFoundError(f"Source must be tar.gz or directory: {source}")

        manifest_path = batch_dir / "batch_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Missing batch_manifest.json in delivery bundle!")

        manifest = BatchManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        ds = dataset_name or manifest.dataset_name
        sp = split_id or manifest.split_id

        console.print(
            f"[bold cyan]Delivery from host:[/bold cyan] {manifest.hostname} | dataset={ds}/{sp}"
        )

        # Layer 1 & 2
        problems, extras = verify_batch_files(batch_dir, manifest)
        computed_hash = manifest.compute_batch_hash()
        layer1_ok = len(problems) == 0

        console.print(
            f"  Layer 1 File Integrity: {'[green]OK[/green]' if layer1_ok else '[red]FAILED[/red]'}"
        )
        if not layer1_ok:
            for p in problems[:5]:
                console.print(f"    [red]{p}[/red]")
            return 2

        # Layers 3 & 4
        prep_dir = paths.artifacts_dir / "preprocessed" / ds / sp
        prep_bundle = (
            PreprocessedBundle.load(prep_dir)
            if (prep_dir / "feature_manifest.json").is_file()
            else None
        )

        run_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
        reports: list[RunAuditReport] = []
        for run_dir in run_dirs:
            report = audit_run_dir(run_dir, prep_bundle=prep_bundle)
            reports.append(report)

        table = Table(title=f"Delivery Audit ({len(reports)} runs)")
        table.add_column("Run", style="cyan")
        table.add_column("Verdict", justify="center")
        table.add_column("Failed Checks")
        for r in reports:
            failed = r.failed_hard_checks + r.failed_soft_checks
            color = (
                "green"
                if r.verdict == AuditVerdict.PASS
                else ("yellow" if r.verdict == AuditVerdict.WARN else "red")
            )
            table.add_row(
                r.run_id,
                f"[{color}]{r.verdict.value.upper()}[/{color}]",
                ", ".join(failed) if failed else "-",
            )
        console.print(table)

        passing = [r for r in reports if r.verdict != AuditVerdict.FAIL]
        failing = [r for r in reports if r.verdict == AuditVerdict.FAIL]

        batch_label = f"{sp}_{computed_hash[:12]}"
        audit_out = paths.audits_dir / "gpu_runs" / batch_label
        log_path = paths.audits_dir / "gpu_runs" / "ingestion_log.jsonl"
        results_root = paths.results_dir / ds / sp

        ingested = []
        quarantined = []
        skipped = []

        if not dry_run:
            audit_out.mkdir(parents=True, exist_ok=True)
            for r in passing:
                src_dir = batch_dir / r.dir_name
                dest = results_root / r.dir_name
                if dest.exists() and not force:
                    skipped.append(r.dir_name)
                    continue
                if dest.exists() and force:
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dir, dest)
                ingested.append(r.dir_name)

            for r in failing:
                qdir = audit_out / "quarantine" / r.dir_name
                qdir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(batch_dir / r.dir_name, qdir)
                quarantined.append(r.dir_name)

            append_ingestion_log(
                log_path,
                {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "batch_hash": computed_hash,
                    "batch_label": batch_label,
                    "source": str(source),
                    "n_runs": len(reports),
                    "ingested": ingested,
                    "quarantined": quarantined,
                    "skipped": skipped,
                },
            )
            console.print(
                f"[bold green]Ingested {len(ingested)} runs into:[/bold green] {results_root}"
            )
            if quarantined:
                console.print(
                    f"[bold red]Quarantined {len(quarantined)} runs under:[/bold red] {audit_out}/quarantine"
                )
        return 0
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.exit(
        receive_delivery(
            source=args.source,
            dataset_name=args.dataset,
            split_id=args.split,
            dry_run=args.dry_run,
            force=args.force,
        )
    )
