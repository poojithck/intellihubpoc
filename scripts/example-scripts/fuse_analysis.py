#!/usr/bin/env python3
"""
Fuse Analysis Script

This script analyzes fuse cartridge images to determine if they have been pulled out.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ImageAnalyzer
from src.config import ConfigManager
from src.utils import setup_logging

async def main():
    """Main function to run the fuse analysis."""
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Setup logging
        setup_logging(config_manager)
        
        # Initialize analyzer with fuse analysis configuration
        analyzer = ImageAnalyzer(config_manager, "fuse_analysis")
        
        # Analyze images (uses configured default path)
        results = await analyzer.analyze_images()
        
        # Display results
        analyzer.display_results(results)
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 