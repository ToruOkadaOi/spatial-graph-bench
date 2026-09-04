"""Runtime version capture for Git commit and dependency libraries."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_code_version(repo_root: Path | None = None) -> str:
    """Get current git commit hash with dirty flag if uncommitted changes exist."""
    try:
        cmd = ["git", "describe", "--always", "--dirty", "--long"]
        if repo_root:
            cmd = ["git", "-C", str(repo_root)] + cmd[1:]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_torch_geometric_version() -> str | None:
    try:
        import torch_geometric

        return str(torch_geometric.__version__)
    except ImportError:
        return None
