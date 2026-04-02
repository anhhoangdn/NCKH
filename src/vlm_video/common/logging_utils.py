"""Logging utilities for vlm_video."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The logger writes to *stderr* at the level specified by the ``LOG_LEVEL``
    environment variable (default ``INFO``).  If *log_file* is provided, an
    additional :class:`logging.FileHandler` is attached.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    log_file:
        Optional path to a log file.  The parent directory is created if needed.

    Returns
    -------
    logging.Logger
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
