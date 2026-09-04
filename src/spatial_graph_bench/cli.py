"""Command line interface for spatial-graph-bench."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="spatial-bench",
        description="Spatial Graph Inductive-Benefit Benchmark for Spatial Transcriptomics",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Census audit
    subparsers.add_parser("census", help="Audit raw data census and verification gates")

    args = parser.parse_args()
    if args.command == "census":
        from scripts.audit_census import audit_census

        ok = audit_census()
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
