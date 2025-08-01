#!/usr/bin/env python3
"""
Switch Installation Analysis Script

Analyzes multiple images from a work order to determine if new switches have been installed.
Uses multi-image analysis to compare before and after images of the meter board.
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
from src.tools.image_gridder import ImageGridder
from src.utils import setup_logging


async def analyse_switch_installation(image_folder: str) -> Dict[str, Any]:
    """Analyse a folder for switch installation using multi-image analysis with image grids."""
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    config_manager = ConfigManager()
    setup_logging(config_manager)
    analyzer = ImageAnalyzer(config_manager, "SwitchInstallation")
    gridder = ImageGridder(config_manager)

    # --- Create grid images ---
    grids = gridder.create_grids(image_folder, output_dir="artefacts/test_grids")
    if not grids:
        raise RuntimeError("No grid images could be created from input images.")

    # --- Encode grid images ---
    encoded_grids = gridder.encode_grids(grids, format="PNG")

    logging.info(f"Loaded {len(encoded_grids)} grid images for multi-image analysis")

    # --- Use multi-image analysis method ---
    prompt_config = config_manager.get_prompt_config("SwitchInstallation")
    model_params = config_manager.get_model_params(config_type="analysis")

    response = analyzer.bedrock_client.invoke_model_multi_image(
        prompt=prompt_config["main_prompt"],
        images=encoded_grids,
        max_tokens=model_params.get("max_tokens", 2000),
        temperature=model_params.get("temperature", 0.1)
    )

    # Parse the response
    response_text = response.get("text", "")
    from src.clients.bedrock_client import BedrockClient
    response_text = BedrockClient.repair_json_response(response_text)

    parsed_response = analyzer.bedrock_client.parse_json_response(
        response_text, 
        analyzer.fallback_parser
    )

    # Extract required fields
    result = {
        "init_image": parsed_response.get("init_image", ""),
        "final_image": parsed_response.get("final_image", ""),
        "switch_installed": parsed_response.get("switch_installed", False),
        "switch_count": parsed_response.get("switch_count", 0),
        "notes": parsed_response.get("notes", "Analysis completed"),
        "total_cost": response.get("total_cost", 0),
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0)
    }

    return result


async def main():
    # Default image directory (matches sample data structure)
    default_dir = Path(__file__).parent.parent / "artefacts" / "Switch-Replacement" / "Example 1"

    # Allow optional CLI argument to override the default directory
    folder_path = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)

    if not Path(folder_path).exists():
        print("Image directory not found. Provide a valid path as an argument.")
        sys.exit(1)

    try:
        result_json = await analyse_switch_installation(folder_path)
        print(json.dumps(result_json, indent=2))
    except Exception as e:
        logging.error(f"Switch installation analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 