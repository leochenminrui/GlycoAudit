"""Logging configuration for the glyco mirror."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: Path,
    level: int = logging.INFO,
    log_to_console: bool = True,
) -> logging.Logger:
    """Set up logging for the mirror.

    Args:
        log_dir: Directory for log files.
        level: Logging level.
        log_to_console: Whether to also log to console.

    Returns:
        Configured logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("glyco_mirror")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler - main log
    log_file = log_dir / "mirror.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # File handler - session-specific log
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log = log_dir / f"mirror_{session_time}.log"
    session_handler = logging.FileHandler(session_log, encoding="utf-8")
    session_handler.setLevel(level)
    session_handler.setFormatter(file_formatter)
    logger.addHandler(session_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional sub-logger name.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"glyco_mirror.{name}")
    return logging.getLogger("glyco_mirror")


class DownloadLogger:
    """Structured logger for download events."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def download_start(self, source: str, item_id: str, url: str) -> None:
        """Log download start."""
        self.logger.info(f"[{source}] Starting download: {item_id} from {url}")

    def download_success(
        self, source: str, item_id: str, local_path: Path, size_bytes: int
    ) -> None:
        """Log successful download."""
        size_mb = size_bytes / (1024 * 1024)
        self.logger.info(
            f"[{source}] Downloaded: {item_id} -> {local_path} ({size_mb:.2f} MB)"
        )

    def download_skip(self, source: str, item_id: str, reason: str) -> None:
        """Log skipped download."""
        self.logger.info(f"[{source}] Skipped: {item_id} - {reason}")

    def download_fail(self, source: str, item_id: str, error: str) -> None:
        """Log failed download."""
        self.logger.error(f"[{source}] Failed: {item_id} - {error}")

    def download_retry(
        self, source: str, item_id: str, attempt: int, max_attempts: int
    ) -> None:
        """Log retry attempt."""
        self.logger.warning(
            f"[{source}] Retrying: {item_id} (attempt {attempt}/{max_attempts})"
        )

    def source_start(self, source: str, total_items: int | None = None) -> None:
        """Log source processing start."""
        if total_items is not None:
            self.logger.info(f"[{source}] Starting processing of {total_items} items")
        else:
            self.logger.info(f"[{source}] Starting processing")

    def source_complete(
        self, source: str, success: int, failed: int, skipped: int
    ) -> None:
        """Log source processing complete."""
        self.logger.info(
            f"[{source}] Complete - success: {success}, failed: {failed}, skipped: {skipped}"
        )
