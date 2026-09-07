"""Structured logging infrastructure with dual Rich console and persistent file logging."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.logging import RichHandler

from spatial_graph_bench.utils.paths import get_project_root

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_log_level(level: int | str | None) -> int:
    """Resolve log level from int, string name, or SPATIAL_LOG_LEVEL env var."""
    if level is None:
        level_str = os.getenv("SPATIAL_LOG_LEVEL", "INFO").upper()
    elif isinstance(level, str):
        level_str = level.upper()
    else:
        return level

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str, logging.INFO)


def setup_logging(
    log_file: Path | str | None = None,
    level: int | str | None = None,
    console_output: bool = True,
    name: str = "spatial_graph_bench",
) -> logging.Logger:
    """Configure structured logging for both terminal and disk persistence.

    Parameters
    ----------
    log_file : Path | str | None
        Target log file path. If None, defaults to `logs/spatial_bench.log` at the project root.
    level : int | str | None
        Logging level (e.g. "DEBUG", "INFO", logging.INFO).
    console_output : bool
        Whether to attach a RichHandler for terminal output.
    name : str
        Logger namespace.
    """
    resolved_level = parse_log_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    # Clear existing handlers to prevent duplication if reconfigured
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler (Rich)
    if console_output:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(resolved_level)
        logger.addHandler(console_handler)

    # 2. File Handler (Persistent disk log)
    if log_file is None:
        target_path = get_project_root() / "logs" / "spatial_bench.log"
    else:
        target_path = Path(log_file).resolve()

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT))
        file_handler.setLevel(resolved_level)
        logger.addHandler(file_handler)
    except OSError:
        # Fallback gracefully if filesystem path is unwritable
        pass

    return logger


def get_logger(
    name: str = "spatial_graph_bench",
    level: int | str | None = None,
) -> logging.Logger:
    """Retrieve an existing logger or initialize one with default dual-stream handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(level=level, name=name)
    if level is not None:
        logger.setLevel(parse_log_level(level))
    return logger
