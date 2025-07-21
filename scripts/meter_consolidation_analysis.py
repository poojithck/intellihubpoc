#!/usr/bin/env python3
"""
Meter Consolidation Analysis Script

Analyzes multiple images from a work order to determine if meter consolidation occurred.
Uses multi-image analysis to compare all images simultaneously.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.tools import ImageAnalyzer
from src.utils import setup_logging


async def analyse_meter_consolidation(image_folder: str) -> Dict[str, Any]:
    """Analyse a folder for meter consolidation using multi-image analysis.

    Returns JSON with required fields.
    """
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    config_manager = ConfigManager()

    # Setup logging using existing utility
    setup_logging(config_manager)

    # Initialize analyzer for loading images
    analyzer = ImageAnalyzer(config_manager, "MeterConsolidationE4")

    # --- Load & encode images (retaining timestamps) ---
    encoded_images = analyzer.load_and_process_images(image_folder)
    if not encoded_images:
        raise RuntimeError("No images found or failed to encode images.")

    logging.info(f"Loaded {len(encoded_images)} images for multi-image analysis")

    # --- Use new multi-image analysis method ---
    # Get prompt configuration
    prompt_config = config_manager.get_prompt_config("MeterConsolidationE4")
    model_params = config_manager.get_model_params(config_type="analysis")

    # Use bedrock client directly for multi-image analysis
    response = analyzer.bedrock_client.invoke_model_multi_image(
        prompt=prompt_config["main_prompt"],
        images=encoded_images,
        max_tokens=model_params.get("max_tokens", 2000),
        temperature=model_params.get("temperature", 0.1)
    )

    # Parse the response
    response_text = response.get("text", "")
    logging.info(f"Raw LLM response: {response_text}")
    
    # Extract and repair truncated JSON
    json_start = response_text.find('{')
    if json_start != -1:
        json_text = response_text[json_start:]
        
        # If truncated (more open braces than close), try to complete it
        if json_text.count('{') > json_text.count('}'):
            # Find last complete field by looking for last comma before truncation
            last_comma = json_text.rfind(',')
            if last_comma > 0:
                json_text = json_text[:last_comma] + '\n}'
            else:
                json_text += '}'
        
        response_text = json_text
        logging.info(f"Repaired JSON: {json_text}")
    
    parsed_response = analyzer.bedrock_client.parse_json_response(
        response_text, 
        analyzer.fallback_parser
    )

    # Extract required fields
    result = {
        "init_count": parsed_response.get("init_count", 0),
        "final_count": parsed_response.get("final_count", 0),
        "init_image": parsed_response.get("init_image", ""),
        "final_image": parsed_response.get("final_image", ""),
        "consolidation": parsed_response.get("consolidation", False),
        "notes": parsed_response.get("notes", "Analysis completed"),
        "total_cost": response.get("total_cost", 0),
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0)
    }

    return result


async def main():
    # Default image directory (matches sample data structure)
    default_dir = Path(__file__).parent.parent / "artefacts" / "Meter Consolidation" / "Example 1"

    # Allow optional CLI argument to override the default directory
    folder_path = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)

    if not Path(folder_path).exists():
        print("Image directory not found. Provide a valid path as an argument.")
        sys.exit(1)

    try:
        result_json = await analyse_meter_consolidation(folder_path)
        print(json.dumps(result_json, indent=2))
    except Exception as e:
        logging.error(f"Meter consolidation analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 