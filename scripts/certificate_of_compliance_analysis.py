#!/usr/bin/env python3
"""
Certificate of Compliance Analysis Script

Analyzes multiple images from a work order to:
- Find a photo or screenshot of a Certificate of Compliance (PDF or hand-filled form)
- Confirm the presence of 'Electrical Work' and form validity
- Return required fields for claim validation
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.tools import ImageAnalyzer
from src.utils import setup_logging
from src.tools.image_gridder import ImageGridder


async def analyse_certificate_of_compliance(image_folder: str) -> Dict[str, Any]:
    """Analyse a folder for a certificate of compliance using multi-image analysis with image grids."""
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    config_manager = ConfigManager()
    setup_logging(config_manager)
    analyzer = ImageAnalyzer(config_manager, "CertificateOfCompliance")
    gridder = ImageGridder(config_manager)

    # --- Create grid images ---
    grids = gridder.create_grids(image_folder, output_dir="artefacts/test_grids")
    if not grids:
        raise RuntimeError("No grid images could be created from input images.")

    # --- Encode grid images ---
    from src.tools.image_loader import ImageLoader
    encoded_grids = []
    for grid_name, grid_img in grids:
        encoded_data = ImageLoader.encode_single_image(None, grid_img, format="PNG")
        encoded_grids.append({
            "name": grid_name,
            "data": encoded_data,
            "timestamp": None
        })

    logging.info(f"Loaded {len(encoded_grids)} grid images for multi-image analysis")

    # --- Use multi-image analysis method ---
    prompt_config = config_manager.get_prompt_config("CertificateOfCompliance")
    model_params = config_manager.get_model_params(config_type="analysis")

    response = analyzer.bedrock_client.invoke_model_multi_image(
        prompt=prompt_config["main_prompt"],
        images=encoded_grids,
        max_tokens=model_params.get("max_tokens", 2000),
        temperature=model_params.get("temperature", 0.1)
    )

    response_text = response.get("text", "")

    # Extract and repair truncated JSON
    json_start = response_text.find('{')
    if json_start != -1:
        json_text = response_text[json_start:]
        if json_text.count('{') > json_text.count('}'):
            last_comma = json_text.rfind(',')
            if last_comma > 0:
                json_text = json_text[:last_comma] + '\n}'
            else:
                json_text += '}'
        response_text = json_text

    parsed_response = analyzer.bedrock_client.parse_json_response(
        response_text,
        analyzer.fallback_parser
    )

    # Extract required fields
    result = {
        "Certificate_image": parsed_response.get("Certificate_image", ""),
        "Certificate_type": parsed_response.get("Certificate_type", "none"),
        "Electrical_Work_Present": parsed_response.get("Electrical_Work_Present", False),
        "Valid_Certificate": parsed_response.get("Valid_Certificate", False),
        "Notes": parsed_response.get("Notes", ""),
        "total_cost": response.get("total_cost", 0),
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0)
    }

    return result


async def main():
    # Default image directory (matches sample data structure)
    default_dir = Path(__file__).parent.parent / "artefacts" / "Certificate of Compliance" / "Example 1"

    # Allow optional CLI argument to override the default directory
    folder_path = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)

    if not Path(folder_path).exists():
        print("Image directory not found. Provide a valid path as an argument.")
        sys.exit(1)

    try:
        result_json = await analyse_certificate_of_compliance(folder_path)
        print(json.dumps(result_json, indent=2))
    except Exception as e:
        logging.error(f"Certificate of compliance analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 