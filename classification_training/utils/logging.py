# utils/logging.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    script_name: str,
    data_dir: Path,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration with rotating file handler and console handler.

    Args:
        script_name: Name of the script (without .py extension) for log filename
        data_dir: Path to data directory where logs folder will be created
        file_level: Logging level for file handler (default: DEBUG)
        console_level: Logging level for console handler (default: INFO)
        max_bytes: Maximum size of each log file in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        format_string: Custom format string for log messages

    Returns:
        Logger instance configured for the calling script

    Example:
        from utils.logging import setup_logging
        from detection_labelling.config import DATA_DIR

        logger = setup_logging("extract_and_upload_frames", DATA_DIR)
        logger.info("Script started")
    """
    # Set up logs directory
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_filename = f"{script_name}.log"
    log_path = log_dir / log_filename

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(file_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Configure root logger
    logging.basicConfig(
        level=min(file_level, console_level),  # Set to lowest level
        format=format_string,
        handlers=[file_handler, console_handler],
        force=True,  # Override any existing configuration
    )

    # Get logger for the calling module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {script_name}")
    logger.info(f"Log file: {log_path}")
    logger.info(
        f"File level: {logging.getLevelName(file_level)}, Console level: {logging.getLevelName(console_level)}"
    )

    return logger
