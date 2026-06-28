"""
utils/logger.py
─────────────────────────────────────────────────────────────────────────────
Centralised Loguru logger with rotating file sink and structured JSON output.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings, BASE_DIR

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _setup_logger() -> None:
    logger.remove()

    # ── Console ───────────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # ── Rotating file (plain text) ────────────────────────────────────────────
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",       # new file each midnight
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,           # thread-safe
    )

    # ── Structured JSON sink (for log-aggregation pipelines) ─────────────────
    logger.add(
        LOG_DIR / "app_structured.jsonl",
        level="WARNING",
        rotation="100 MB",
        retention="90 days",
        compression="gz",
        serialize=True,         # writes newline-delimited JSON
        enqueue=True,
    )


_setup_logger()

__all__ = ["logger"]
