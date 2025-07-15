#!/usr/bin/env python3
"""
Meter Reading Analysis Script

Analyzes utility meter images to determine meter readings.
Processes images from artefacts/Meter Reads/ folder.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ConfigurableAnalyzer
from src.config import ConfigManager


async def main():
    """Main function to run meter reading analysis."""
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Initialize analyzer with meter reading configuration
    analyzer = ConfigurableAnalyzer(config_manager, "meter_reading")
    
    # Set up image directory
    image_dir = Path(__file__).parent.parent / "artefacts" / "Meter Reads"
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    print("🔍 Starting Meter Reading Analysis...")
    print(f"📁 Processing images from: {image_dir}")
    print("-" * 50)
    
    # Run analysis
    results = await analyzer.analyze_images(str(image_dir))
    
    # Display results
    analyzer.display_results(results, "Meter Reading Analysis")


if __name__ == "__main__":
    asyncio.run(main()) 