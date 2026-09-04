"""Logging utilities using Rich for clean terminal formatting."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str = "spatial_graph_bench", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
