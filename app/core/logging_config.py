"""
logging_config.py
-----------------
Basic logging configuration for the Green-Cycle API.

Provides consistent log formatting across all modules.
"""

import logging
import os


def setup_logging() -> None:
    """
    Configure application-wide logging.

    Log level can be controlled using the LOG_LEVEL environment variable.
    Default level: INFO
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
