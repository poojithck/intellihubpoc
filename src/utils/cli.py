"""
CLI utility functions for scripts.
"""

import logging
from ..config import ConfigManager


def setup_logging(config_manager: ConfigManager) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config_manager: ConfigManager instance to get logging configuration
    """
    logging_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ) 