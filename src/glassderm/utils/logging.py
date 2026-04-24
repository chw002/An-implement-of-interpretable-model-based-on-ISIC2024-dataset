"""Single logger factory so every CLI sub-command writes to a consistent place."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s"


def get_logger(name: str = "glassderm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def setup_file_logging(logger: logging.Logger, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
