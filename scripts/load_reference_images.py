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


def determine_category_from_filename(filename: str, sor_type: str) -> str:
    """
    Determine category based on filename pattern.
    
    Patterns for FuseReplacement:
    - valid_fuse_* -> valid_fuse
    - not_valid_fuse_* -> not_valid_fuse
    - invalid_fuse_* -> not_valid_fuse
    
    Patterns for MeterConsolidationE4:
    - valid_meter_* -> valid_meter
    - not_meter_* -> not_a_meter
    """
    filename_lower = filename.lower()
    
    if sor_type == "FuseReplacement":
        if filename_lower.startswith("valid_fuse"):
            return "valid_fuse"
        elif filename_lower.startswith("not_valid_fuse") or filename_lower.startswith("not_fuse"):
            return "not_valid_fuse"
        elif filename_lower.startswith("invalid_fuse"):
            return "not_valid_fuse"
        else:
            # Default for fuses based on pattern
            print(f"Warning: Could not determine category for {filename}, defaulting to valid_fuse")
            return "valid_fuse"
    
    elif sor_type == "MeterConsolidationE4":
        if filename_lower.startswith("valid_meter"):
            return "valid_meter"
        elif filename_lower.startswith("not_meter") or filename_lower.startswith("not_a_meter"):
            return "not_a_meter"
        else:
            print(f"Warning: Could not determine category for {filename}, defaulting to valid_meter")
            return "valid_meter"
    
    else:
        print(f"Warning: Unknown SOR type {sor_type}, defaulting to valid_meter category")
        return "valid_meter"


def main():
    parser = argparse.ArgumentParser(description="Load reference images into RAG repository")
    parser.add_argument("--sor-type", required=True, 
                       choices=["MeterConsolidationE4", "FuseReplacement", "PlugInMeterRemoval", 
                               "ServiceProtectionDevices", "SwitchInstallation", "NeutralLinkInstallation",
                               "AsbestosBagAndBoard", "CertificateOfCompliance"],
                       help="SOR type to load images for")
    parser.add_argument("--clear", action="store_true", help="Clear existing repository before loading")
    parser.add_argument("--regenerate-metadata", action="store_true", 
                       help="Force regenerate metadata.json from filenames")
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
    
    # Check for metadata.json
    metadata_file = repository.working_path / "metadata.json"
    metadata_map = {}
    
    # Load or regenerate metadata
    if metadata_file.exists() and not args.regenerate_metadata:
        print(f"Found existing metadata.json")
        with open(metadata_file, 'r') as f:
            metadata_map = json.load(f)
        
        # Check if metadata has correct categories for FuseReplacement
        if args.sor_type == "FuseReplacement":
            needs_regeneration = False
            for key, value in metadata_map.items():
                if value.get('category') not in ['valid_fuse', 'not_valid_fuse']:
                    print(f"WARNING: Found incorrect category '{value.get('category')}' for {key}")
                    needs_regeneration = True
                    break
            
            if needs_regeneration:
                print("Metadata has incorrect categories. Regenerating...")
                args.regenerate_metadata = True
    
    if args.regenerate_metadata or not metadata_file.exists():
        print(f"Generating metadata.json from filenames...")
        metadata_map = {}
    
    # Find all unique image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = set()  # Use set to avoid duplicates
    
    for ext in image_extensions:
        for pattern in [f'*{ext}', f'*{ext.upper()}']:
            for file in repository.working_path.glob(pattern):
                image_files.add(file)
    
    image_files = sorted(list(image_files))  # Convert back to sorted list
    print(f"Found {len(image_files)} unique image files in {repository.working_path}")
    
    # If regenerating metadata, create it now
    if args.regenerate_metadata or not metadata_map:
        metadata_map = {}
        for image_file in image_files:
            filename_key = image_file.stem
            category = determine_category_from_filename(filename_key, args.sor_type)
            metadata_map[filename_key] = {"category": category}
        
        # Save the regenerated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata_map, f, indent=2)
        print(f"Saved regenerated metadata.json with {len(metadata_map)} entries")
    
    # Process images
    loaded_count = 0
    categories_count = {}
    processed_files = set()  # Track processed files to avoid duplicates
    
    for image_file in image_files:
        # Skip if already processed (handles case-sensitive duplicates)
        if image_file.name.lower() in processed_files:
            continue
        processed_files.add(image_file.name.lower())
        
        filename_key = image_file.stem
        
        # Get category from metadata
        if filename_key in metadata_map:
            category = metadata_map[filename_key].get('category')
        else:
            # Auto-detect from filename if not in metadata
            category = determine_category_from_filename(filename_key, args.sor_type)
            print(f"Auto-detected category for {filename_key}: {category}")
        
        # Add the reference image
        try:
            ref_image = rag_pipeline.add_reference_image(
                str(image_file),
                category=category,
                sor_type=args.sor_type,
                image_id=filename_key
            )
            print(f"Added: {ref_image.id} ({category}) - {image_file.name}")
            loaded_count += 1
            
            # Track categories
            categories_count[category] = categories_count.get(category, 0) + 1
            
        except Exception as e:
            print(f"Error adding {image_file.name}: {e}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Loaded {loaded_count} reference images for {args.sor_type}")
    print(f"\nBreakdown by category:")
    for category, count in sorted(categories_count.items()):
        print(f"  {category}: {count}")
    
    # Get overall statistics
    stats = rag_pipeline.get_statistics(args.sor_type)
    print(f"\nRepository Statistics for {args.sor_type}:")
    print(f"  Total images in repository: {stats['total_images']}")
    
    if args.sor_type == "FuseReplacement":
        print(f"  Valid fuses: {stats.get('valid_fuses', 0)}")
        print(f"  Not valid fuses: {stats.get('not_valid_fuses', 0)}")
        
        # Verify categories are correct
        if stats.get('valid_fuses', 0) == 0 and stats.get('valid_meters', 0) > 0:
            print("\n WARNING: Images are categorized as meters instead of fuses!")
            print("    Run with --clear --regenerate-metadata to fix this.")
    else:
        print(f"  Valid meters: {stats.get('valid_meters', 0)}")
        print(f"  Not meters: {stats.get('not_meters', 0)}")


if __name__ == "__main__":
    main()