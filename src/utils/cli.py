"""
CLI utility functions for scripts.
"""

import logging
from typing import Dict, Any
from ..config import ConfigManager


def setup_logging(config_manager) -> None:
    """Setup logging configuration from config manager."""
    logging_config = config_manager.get_logging_config()
    
    logging.basicConfig(
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ) 