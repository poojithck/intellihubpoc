#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.rag.pipeline import RAGPipeline
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Load reference images into RAG repository")
    parser.add_argument("--sor-type", required=True, 
                       choices=["MeterConsolidationE4", "FuseReplacement", "PlugInMeterRemoval", 
                               "ServiceProtectionDevices", "SwitchInstallation", "NeutralLinkInstallation",
                               "AsbestosBagAndBoard", "CertificateOfCompliance"],
                       help="SOR type to load images for")
    parser.add_argument("--clear", action="store_true", help="Clear existing repository before loading")
    args = parser.parse_args()
    
    # Initialize
    config_manager = ConfigManager()
    setup_logging(config_manager)
    rag_pipeline = RAGPipeline(config_manager)
    
    # Get repository for the specified SOR type
    repository = rag_pipeline.get_repository(args.sor_type)
    if not repository:
        print(f"Error: No repository configured for {args.sor_type}")
        sys.exit(1)
    
    # Clear repository if requested
    if args.clear:
        print(f"Clearing existing repository for {args.sor_type}...")
        for image_id in list(repository.images.keys()):
            repository.remove_image(image_id)
    
    # Load metadata from the SOR-specific directory
    metadata_file = repository.working_path / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {repository.working_path}")
        print(f"Please create {metadata_file} with image metadata")
        sys.exit(1)
    
    with open(metadata_file, 'r') as f:
        metadata_map = json.load(f)
    
    # Process images
    loaded_count = 0
    
    for filename_key, metadata in metadata_map.items():
        # Try different extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = repository.working_path / f"{filename_key}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file:
            category = metadata['category']
            ref_image = rag_pipeline.add_reference_image(
                str(image_file),
                category=category,
                sor_type=args.sor_type,
                image_id=filename_key
            )
            print(f"Added: {ref_image.id} ({category}) - {image_file.name}")
            loaded_count += 1
        else:
            print(f"Warning: No image file found for {filename_key}")
    
    # Print statistics
    print(f"\nLoaded {loaded_count} reference images for {args.sor_type}")
    stats = rag_pipeline.get_statistics(args.sor_type)
    print(f"\nRepository Statistics for {args.sor_type}:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Valid meters: {stats['valid_meters']}")
    print(f"  Not meters: {stats['not_meters']}")


if __name__ == "__main__":
    main()