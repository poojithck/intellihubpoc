"""
CLI utility functions for scripts.
"""

import logging
from typing import Dict, Any
from ..config import ConfigManager


def setup_logging(config_manager) -> None:
    """Setup logging configuration from config manager."""
    logging_config = config_manager.get_logging_config()
    
    # Set main logging level
    main_level = getattr(logging, logging_config.get("level", "INFO"))
    logging.basicConfig(
        level=main_level,
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Apply specific logger level controls to reduce verbose output
    logger_levels = logging_config.get("logger_levels", {})
    for logger_name, level_name in logger_levels.items():
        try:
            level = getattr(logging, level_name.upper())
            logging.getLogger(logger_name).setLevel(level)
        except (AttributeError, ValueError) as e:
            # Fallback to WARNING if invalid level
            logging.getLogger(logger_name).setLevel(logging.WARNING) 