#!/usr/bin/env python3
"""
Meter Reading Analysis Script

Analyzes utility meter images to determine meter readings.
Processes images from artefacts/Meter Reads/ folder.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ImageAnalyzer
from src.config import ConfigManager
from src.utils import setup_logging


async def main():
    """Main function to run meter reading analysis."""
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Setup logging
    setup_logging(config_manager)
    
    # Initialize analyzer with meter reading configuration
    analyzer = ImageAnalyzer(config_manager, "meter_reading")
    
    # Set up image directory
    image_dir = Path(__file__).parent.parent / "artefacts" / "Meter Reads"
    
    if not image_dir.exists():
        logging.error(f"Directory {image_dir} does not exist")
        return
    
    logging.info("🔍 Starting Meter Reading Analysis...")
    logging.info(f"📁 Processing images from: {image_dir}")
    
    # Run analysis
    results = await analyzer.analyze_images(str(image_dir))
    
    # Display results
    analyzer.display_results(results)


if __name__ == "__main__":
    asyncio.run(main()) 