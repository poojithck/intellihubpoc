#!/usr/bin/env python3
"""
Multi-SOR Analysis Script

Analyzes multiple images from a work order using all available SOR prompts.
Sends gridded images once, then iteratively applies each SOR prompt.
Returns comprehensive results for all SOR types.
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


async def analyse_multi_sor(image_folder: str) -> Dict[str, Any]:
    """Analyse a folder using all available SOR prompts with shared gridded images."""
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    config_manager = ConfigManager()
    setup_logging(config_manager)
    gridder = ImageGridder(config_manager)

    # Define all available SOR types
    sor_types = [
        "AsbestosBagAndBoard",
        "CertificateOfCompliance", 
        "FuseReplacement",
        "MeterConsolidationE4",
        "PlugInMeterRemoval",
        "ServiceProtectionDevices",
        "SwitchInstallation",
        "NeutralLinkInstallation"
    ]

    # Create grid images once
    grids = gridder.create_grids(image_folder, output_dir="artefacts/test_grids")
    if not grids:
        raise RuntimeError("No grid images could be created from input images.")

    encoded_grids = gridder.encode_grids(grids, format="PNG")
    logging.info(f"Created {len(encoded_grids)} grid images for multi-SOR analysis")

    # Analyze each SOR type
    results = {}
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for sor_type in sor_types:
        try:
            logging.info(f"Analyzing SOR: {sor_type}")
            
            # Create analyzer for this SOR type
            analyzer = ImageAnalyzer(config_manager, sor_type)
            
            # Get prompt configuration
            prompt_config = config_manager.get_prompt_config(sor_type)
            model_params = config_manager.get_model_params(config_type="analysis")

            # Use multi-image analysis with shared encoded grids
            response = analyzer.bedrock_client.invoke_model_multi_image(
                prompt=prompt_config["main_prompt"],
                images=encoded_grids,
                max_tokens=model_params.get("max_tokens", 2000),
                temperature=model_params.get("temperature", 0.1)
            )

            # Parse response
            response_text = response.get("text", "")
            from src.clients.bedrock_client import BedrockClient
            response_text = BedrockClient.repair_json_response(response_text)

            parsed_response = analyzer.bedrock_client.parse_json_response(
                response_text,
                analyzer.fallback_parser
            )

            # Store results
            results[sor_type] = {
                **parsed_response,
                "total_cost": response.get("total_cost", 0),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0)
            }

            # Accumulate costs
            total_cost += response.get("total_cost", 0)
            total_input_tokens += response.get("input_tokens", 0)
            total_output_tokens += response.get("output_tokens", 0)

            logging.info(f"Completed {sor_type} analysis")

        except Exception as e:
            logging.error(f"Failed to analyze {sor_type}: {e}")
            results[sor_type] = {
                "error": str(e),
                "total_cost": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

    # Create final result
    result = {
        "work_order": folder.name,
        "sor_results": results,
        "summary": {
            "total_sors": len(sor_types),
            "successful_analyses": len([r for r in results.values() if "error" not in r]),
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }
    }

    return result


async def main():
    # Default image directory
    default_dir = Path(__file__).parent.parent / "artefacts" / "Combined SORS" / "Example 1"

    # Allow optional CLI argument to override the default directory
    folder_path = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)

    if not Path(folder_path).exists():
        print("Image directory not found. Provide a valid path as an argument.")
        sys.exit(1)

    try:
        result_json = await analyse_multi_sor(folder_path)
        print(json.dumps(result_json, indent=2))
    except Exception as e:
        logging.error(f"Multi-SOR analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 