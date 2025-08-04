"""CLI Configuration Utility for Unified SOR Processor."""

import argparse
import sys
from typing import Any, Dict, Optional
from pathlib import Path


class CLIConfig:
    """Handles command line argument parsing and configuration merging."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.sor_config = config_manager.get_config("sor_analysis_config")
        self.cli_defaults = self.sor_config.get("sor_analysis", {}).get("cli_defaults", {})
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Unified SOR Processor - Efficient batch processing of work orders",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process all work orders with all SOR types (uses config default folder)
  python unified_sor_processor.py
  
  # Test mode: Process limited work orders (uses config default limit)
  python unified_sor_processor.py --test-mode
  
  # Process specific number of work orders
  python unified_sor_processor.py --max-work-orders 10
  
  # Process with custom folder
  python unified_sor_processor.py --parent-folder path/to/work_orders
  
  # Process with specific SOR types only
  python unified_sor_processor.py --sor-list AsbestosBagAndBoard,FuseReplacement
  
  # Process with Excel output
  python unified_sor_processor.py --output-format excel
  
  # Process with custom output path
  python unified_sor_processor.py --output-path results/batch_analysis
  
  # List available SOR types
  python unified_sor_processor.py --list-sors
            """
        )
        
        # Optional arguments
        parser.add_argument(
            "--parent-folder",
            help="Path to parent folder containing work order sub-folders (default: uses config default)"
        )
        
        parser.add_argument(
            "--sor-list",
            help="Comma-separated list of specific SOR types to analyze (default: all enabled)"
        )
        
        parser.add_argument(
            "--output-format",
            choices=["csv", "excel", "both"],
            default=self.cli_defaults.get("output_format", "both"),
            help="Output format for table (default: from config)"
        )
        
        parser.add_argument(
            "--output-path",
            help="Base path for output files (default: uses config default with timestamp)"
        )
        
        parser.add_argument(
            "--test-mode",
            action="store_true",
            default=self.cli_defaults.get("test_mode", False),
            help="Run in test mode with limited work orders (uses config default)"
        )
        
        parser.add_argument(
            "--max-work-orders",
            type=int,
            help="Maximum number of work orders to process (overrides test mode)"
        )
        
        parser.add_argument(
            "--list-sors",
            action="store_true",
            help="List all available SOR types and exit"
        )
        
        parser.add_argument(
            "--save-json",
            action="store_true",
            default=self.cli_defaults.get("save_json", True),
            help="Save detailed JSON results (default: from config)"
        )
        
        return parser
    
    def resolve_config(self, args) -> Dict[str, Any]:
        """Resolve final configuration by merging CLI args with config defaults."""
        # Determine parent folder (CLI arg overrides config default)
        parent_folder = (args.parent_folder or 
                        self.cli_defaults.get("parent_folder") or
                        self.sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_parent_folder"))
        
        if not parent_folder:
            print("Error: No parent folder specified and no default found in config")
            print("Use --parent-folder argument or set batch_parent_folder in sor_analysis_config.yaml")
            sys.exit(1)
        
        # Determine SOR types to analyze
        sor_types = None
        if args.sor_list:
            sor_types = [s.strip() for s in args.sor_list.split(",")]
        elif self.cli_defaults.get("sor_list"):
            sor_types = self.cli_defaults.get("sor_list")
        
        # Determine output path
        output_path = args.output_path or self.cli_defaults.get("output_path")
        if not output_path:
            # Use default from config with timestamp
            from datetime import datetime
            default_output = self.sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_output_folder", "outputs/batch_results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{default_output}_{timestamp}"
        
        # Determine work order limit (CLI > config defaults)
        max_work_orders = (args.max_work_orders or 
                          self.cli_defaults.get("max_work_orders"))
        
        if args.test_mode or self.cli_defaults.get("test_mode"):
            test_limit = self.sor_config.get("sor_analysis", {}).get("batch_processing", {}).get("test_max_work_orders", 5)
            max_work_orders = max_work_orders or test_limit
        
        return {
            "parent_folder": parent_folder,
            "sor_types": sor_types,
            "output_path": output_path,
            "max_work_orders": max_work_orders,
            "output_format": args.output_format,
            "save_json": args.save_json,
            "test_mode": args.test_mode or self.cli_defaults.get("test_mode", False),
            "list_sors": args.list_sors
        }
    
    def print_processing_info(self, config: Dict[str, Any]):
        """Print processing information to console."""
        print(f"Starting unified SOR processing...")
        print(f"Parent folder: {config['parent_folder']}")
        print(f"SOR types: {config['sor_types'] if config['sor_types'] else 'All enabled'}")
        print(f"Processing mode: Parallel")
        print(f"Output path: {config['output_path']}")
        
        if config['max_work_orders']:
            if config['test_mode']:
                print(f"Test mode: Limiting to {config['max_work_orders']} work orders")
            else:
                print(f"Limiting to {config['max_work_orders']} work orders")
        else:
            print("Processing all available work orders")
        
        print("-" * 60)