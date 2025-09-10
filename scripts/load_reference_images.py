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
    parser.add_argument("--image-dir", default="data/reference_images", help="Directory containing reference images")
    parser.add_argument("--clear", action="store_true", help="Clear existing repository before loading")
    args = parser.parse_args()
    
    # Initialize
    config_manager = ConfigManager()
    setup_logging(config_manager)
    rag_pipeline = RAGPipeline(config_manager)
    
    # Clear repository if requested
    if args.clear:
        print("Clearing existing repository...")
        for image_id in list(rag_pipeline.repository.images.keys()):
            rag_pipeline.repository.remove_image(image_id)
    
    # Load metadata
    image_dir = Path(args.image_dir)
    metadata_file = image_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {image_dir}")
        sys.exit(1)
    
    with open(metadata_file, 'r') as f:
        metadata_map = json.load(f)
    
    # Process images
    loaded_count = 0
    
    for filename_key, metadata in metadata_map.items():
        # Try different extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = image_dir / f"{filename_key}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if image_file:
            category = metadata['category']
            ref_image = rag_pipeline.add_reference_image(
                str(image_file),
                category=category,
                image_id=filename_key
            )
            print(f"Added: {ref_image.id} ({category}) - {image_file.name}")
            loaded_count += 1
        else:
            print(f"Warning: No image file found for {filename_key}")
    
    # Print statistics
    print(f"\nLoaded {loaded_count} reference images")
    stats = rag_pipeline.get_statistics()
    print("\nRepository Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Valid meters: {stats['valid_meters']}")
    print(f"  Not meters: {stats['not_meters']}")


if __name__ == "__main__":
    main()