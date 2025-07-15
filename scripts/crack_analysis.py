#!/usr/bin/env python3
"""
Crack Analysis Script

This script analyzes images to detect cracks longer than 5mm in surfaces.
Demonstrates how easy it is to create new analysis types using the tools.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ImageAnalyzer
from src.config import ConfigManager

def setup_logging(config_manager: ConfigManager) -> None:
    """Setup logging based on configuration."""
    logging_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

async def main():
    """Main function to run the crack analysis."""
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Setup logging
        setup_logging(config_manager)
        
        # Initialize analyzer with crack analysis configuration
        analyzer = ImageAnalyzer(config_manager, "crack_analysis")
        
        # You can specify a custom image folder or use the default
        # analyzer.analyze_images("path/to/crack/images")
        
        # For demo purposes, we'll use the default folder
        # (you would put crack images in the configured default folder)
        results = await analyzer.analyze_images()
        
        # Display results
        analyzer.display_results(results)
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 