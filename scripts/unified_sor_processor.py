#!/usr/bin/env python3
"""
Unified SOR Processor

A comprehensive system that efficiently processes multiple work orders and SOR types.
Key features:
- Discovers work orders in parent folder
- Generates single grid per work order (efficient)
- Processes all SOR types using shared grids
- Generates comprehensive output (tables, summaries, JSON)

Usage:
    # Process all work orders with all SOR types
    python unified_sor_processor.py --parent-folder path/to/work_orders
    
    # Process with specific SOR types only
    python unified_sor_processor.py --parent-folder path/to/work_orders --sor-list AsbestosBagAndBoard,FuseReplacement
    
    # Process with custom output format
    python unified_sor_processor.py --parent-folder path/to/work_orders --output-format excel
    
    # Process with parallel processing disabled
    python unified_sor_processor.py --parent-folder path/to/work_orders --sequential
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.tools import WorkOrderProcessor, ResultsTableGenerator
from src.tools.image_gridder import ImageGridder
from src.utils import setup_logging


class UnifiedSORProcessor:
    """
    Unified processor for efficient SOR analysis across multiple work orders.
    
    Key optimizations:
    - Single grid generation per work order
    - Shared grid usage across all SOR types
    - Efficient batch processing
    - Comprehensive output generation
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.sor_config = config_manager.get_config("sor_analysis_config")
        self.work_order_processor = WorkOrderProcessor(config_manager)
        self.table_generator = ResultsTableGenerator(config_manager)
        self.gridder = ImageGridder(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.batch_config = self.sor_config.get("sor_analysis", {}).get("batch_processing", {})
        self.default_settings = self.sor_config.get("sor_analysis", {}).get("default_settings", {})
        
    def get_enabled_sor_types(self) -> List[str]:
        """Get list of enabled SOR types from configuration."""
        sor_types = self.sor_config.get("sor_analysis", {}).get("sor_types", {})
        return [sor_type for sor_type, config in sor_types.items() 
                if config.get("enabled", True)]
    
    def validate_sor_types(self, sor_types: List[str]) -> List[str]:
        """Validate SOR types and return only valid ones."""
        available_sors = self.get_enabled_sor_types()
        valid_sors = [sor for sor in sor_types if sor in available_sors]
        invalid_sors = [sor for sor in sor_types if sor not in available_sors]
        
        if invalid_sors:
            self.logger.warning(f"Invalid SOR types ignored: {invalid_sors}")
        
        return valid_sors
    
    async def process_single_work_order(self, work_order_info: Dict[str, Any], 
                                      sor_types: List[str]) -> Dict[str, Any]:
        """
        Process a single work order with all specified SOR types.
        
        Optimizations:
        - Generate grid once per work order
        - Share encoded grids across all SOR types
        - Efficient memory usage
        """
        work_order_path = work_order_info["path"]
        work_order_number = work_order_info["work_order_number"]
        
        self.logger.info(f"Processing work order {work_order_number} with {len(sor_types)} SOR types")
        
        # Generate grid images once for this work order
        grid_output_dir = self.default_settings.get("grid_output_dir", "artefacts/test_grids")
        grids = self.gridder.create_grids(work_order_path, output_dir=grid_output_dir)
        
        if not grids:
            raise RuntimeError(f"No grid images could be created for work order {work_order_number}")
        
        # Encode grids once
        output_format = self.default_settings.get("output_format", "PNG")
        encoded_grids = self.gridder.encode_grids(grids, format=output_format)
        
        self.logger.info(f"Created {len(encoded_grids)} grid images for work order {work_order_number}")
        
        # Process all SOR types using the same encoded grids
        results = {}
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for sor_type in sor_types:
            try:
                result = await self._analyze_sor_type(sor_type, work_order_path, encoded_grids, work_order_info)
                results[sor_type] = result
                
                # Accumulate costs
                total_cost += result.get("total_cost", 0)
                total_input_tokens += result.get("input_tokens", 0)
                total_output_tokens += result.get("output_tokens", 0)
                
                self.logger.info(f"Completed {sor_type} analysis for work order {work_order_number}")
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {sor_type} for work order {work_order_number}: {e}")
                results[sor_type] = {
                    "sor_type": sor_type,
                    "work_order": work_order_number,
                    "error": str(e),
                    "total_cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
        
        # Create work order summary
        summary = {
            "work_order": work_order_number,
            "folder_name": work_order_info["folder_name"],
            "image_count": work_order_info["image_count"],
            "analysis_timestamp": datetime.now().isoformat(),
            "total_sors": len(sor_types),
            "successful_analyses": len([r for r in results.values() if "error" not in r]),
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }
        
        return {
            "summary": summary,
            "sor_results": results
        }
    
    async def _analyze_sor_type(self, sor_type: str, work_order_path: str, 
                               encoded_grids: List[bytes], work_order_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single SOR type using pre-encoded grids."""
        from src.tools import ImageAnalyzer
        
        # Create analyzer for this SOR type
        analyzer = ImageAnalyzer(self.config_manager, sor_type)
        
        # Get prompt configuration
        prompt_config = self.config_manager.get_prompt_config(sor_type)
        model_params = self.config_manager.get_model_params(config_type=self.default_settings.get("model_config", "analysis"))
        
        # Use multi-image analysis with pre-encoded grids
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
        
        # Create result with metadata
        result = {
            "sor_type": sor_type,
            "work_order": work_order_info["work_order_number"],
            "folder_name": work_order_info["folder_name"],
            "analysis_timestamp": datetime.now().isoformat(),
            **parsed_response,
            "total_cost": response.get("total_cost", 0),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0)
        }
        
        return result
    
    async def process_work_orders_batch(self, parent_folder: str, 
                                      sor_types: Optional[List[str]] = None,
                                      sequential: bool = False) -> Dict[str, Any]:
        """
        Process multiple work orders in batch with efficient grid generation.
        
        Args:
            parent_folder: Path to parent folder containing work order sub-folders
            sor_types: List of SOR types to analyze (None = all enabled)
            sequential: If True, process work orders sequentially instead of parallel
            
        Returns:
            Dict with batch results and summary
        """
        # Determine SOR types to analyze
        if sor_types is None:
            sor_types = self.get_enabled_sor_types()
        else:
            sor_types = self.validate_sor_types(sor_types)
        
        if not sor_types:
            raise ValueError("No valid SOR types to analyze")
        
        self.logger.info(f"Processing {len(sor_types)} SOR types: {', '.join(sor_types)}")
        
        # Discover work orders
        work_orders = self.work_order_processor.discover_work_orders(parent_folder)
        
        if not work_orders:
            raise ValueError(f"No valid work orders found in: {parent_folder}")
        
        self.logger.info(f"Found {len(work_orders)} work orders to process")
        
        # Process work orders
        if sequential or not self.batch_config.get("parallel_processing", True):
            results = await self._process_work_orders_sequential(work_orders, sor_types)
        else:
            results = await self._process_work_orders_parallel(work_orders, sor_types)
        
        # Create batch summary
        batch_summary = self._create_batch_summary(work_orders, results, sor_types)
        
        return {
            "batch_summary": batch_summary,
            "work_order_results": results
        }
    
    async def _process_work_orders_sequential(self, work_orders: List[Dict[str, Any]], 
                                            sor_types: List[str]) -> Dict[str, Any]:
        """Process work orders sequentially."""
        results = {}
        
        for work_order in work_orders:
            try:
                work_order_number = work_order["work_order_number"]
                self.logger.info(f"Processing work order {work_order_number}")
                
                result = await self.process_single_work_order(work_order, sor_types)
                results[work_order_number] = result
                
            except Exception as e:
                self.logger.error(f"Failed to process work order {work_order['work_order_number']}: {e}")
                results[work_order["work_order_number"]] = {
                    "summary": {
                        "work_order": work_order["work_order_number"],
                        "error": str(e)
                    },
                    "sor_results": {}
                }
        
        return results
    
    async def _process_work_orders_parallel(self, work_orders: List[Dict[str, Any]], 
                                          sor_types: List[str]) -> Dict[str, Any]:
        """Process work orders in parallel with concurrency control."""
        max_concurrent = self.batch_config.get("max_concurrent_work_orders", 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_work_order(work_order: Dict[str, Any]) -> tuple[str, Any]:
            async with semaphore:
                try:
                    work_order_number = work_order["work_order_number"]
                    self.logger.info(f"Processing work order {work_order_number}")
                    result = await self.process_single_work_order(work_order, sor_types)
                    return work_order_number, result
                except Exception as e:
                    self.logger.error(f"Failed to process work order {work_order['work_order_number']}: {e}")
                    return work_order["work_order_number"], {
                        "summary": {
                            "work_order": work_order["work_order_number"],
                            "error": str(e)
                        },
                        "sor_results": {}
                    }
        
        # Create tasks for all work orders
        tasks = [process_single_work_order(wo) for wo in work_orders]
        
        # Execute all tasks
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to dict
        results = {}
        for result in results_list:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
            else:
                work_order_number, work_order_result = result
                results[work_order_number] = work_order_result
        
        return results
    
    def _create_batch_summary(self, work_orders: List[Dict[str, Any]], 
                            results: Dict[str, Any], sor_types: List[str]) -> Dict[str, Any]:
        """Create comprehensive batch summary."""
        total_work_orders = len(work_orders)
        successful_work_orders = len([r for r in results.values() 
                                    if "error" not in r.get("summary", {})])
        failed_work_orders = total_work_orders - successful_work_orders
        
        total_images = sum(wo["image_count"] for wo in work_orders)
        total_cost = sum(r.get("summary", {}).get("total_cost", 0) for r in results.values())
        total_input_tokens = sum(r.get("summary", {}).get("total_input_tokens", 0) for r in results.values())
        total_output_tokens = sum(r.get("summary", {}).get("total_output_tokens", 0) for r in results.values())
        
        # SOR-specific statistics
        sor_statistics = {}
        for sor_type in sor_types:
            pass_count = 0
            fail_count = 0
            error_count = 0
            
            for work_order_result in results.values():
                sor_result = work_order_result.get("sor_results", {}).get(sor_type, {})
                if "error" in sor_result:
                    error_count += 1
                elif sor_result.get("valid_claim", sor_result.get("valid_installation", 
                                                                 sor_result.get("valid_consolidation", 
                                                                               sor_result.get("valid_removal", False)))):
                    pass_count += 1
                else:
                    fail_count += 1
            
            sor_statistics[sor_type] = {
                "pass": pass_count,
                "fail": fail_count,
                "error": error_count
            }
        
        return {
            "processing_timestamp": datetime.now().isoformat(),
            "total_work_orders": total_work_orders,
            "successful_work_orders": successful_work_orders,
            "failed_work_orders": failed_work_orders,
            "total_images": total_images,
            "sor_types_analyzed": sor_types,
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "sor_statistics": sor_statistics,
            "processing_mode": "sequential" if not self.batch_config.get("parallel_processing", True) else "parallel",
            "max_concurrent": self.batch_config.get("max_concurrent_work_orders", 3)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> Dict[str, str]:
        """Save results in multiple formats."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save detailed JSON results
        json_file = output_dir / "batch_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files["json"] = str(json_file)
        
        # Generate and save table
        try:
            df = self.table_generator.generate_table_from_batch_results(results)
            if not df.empty:
                # Save CSV
                csv_file = self.table_generator.save_table(df, str(output_dir / "sor_results"), "csv")
                saved_files["csv"] = csv_file
                
                # Save Excel
                excel_file = self.table_generator.save_table(df, str(output_dir / "sor_results"), "excel")
                saved_files["excel"] = excel_file
                
                # Save summary report
                summary = self.table_generator.generate_summary_statistics(df)
                summary_file = self.table_generator.save_summary_report(summary, str(output_dir / "summary"))
                saved_files["summary"] = summary_file
        except Exception as e:
            self.logger.error(f"Failed to generate tables: {e}")
        
        return saved_files


async def main():
    """Main function with comprehensive command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified SOR Processor - Efficient batch processing of work orders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all work orders with all SOR types
  python unified_sor_processor.py --parent-folder artefacts/work_orders
  
  # Process with specific SOR types only
  python unified_sor_processor.py --parent-folder artefacts/work_orders --sor-list AsbestosBagAndBoard,FuseReplacement
  
  # Process with Excel output and sequential processing
  python unified_sor_processor.py --parent-folder artefacts/work_orders --output-format excel --sequential
  
  # Process with custom output path
  python unified_sor_processor.py --parent-folder artefacts/work_orders --output-path results/batch_analysis
  
  # List available SOR types
  python unified_sor_processor.py --list-sors
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--parent-folder",
        help="Path to parent folder containing work order sub-folders"
    )
    
    # Optional arguments
    parser.add_argument(
        "--sor-list",
        help="Comma-separated list of specific SOR types to analyze (default: all enabled)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["csv", "excel", "both"],
        default="both",
        help="Output format for table (default: both)"
    )
    
    parser.add_argument(
        "--output-path",
        help="Base path for output files (default: uses config default with timestamp)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process work orders sequentially instead of parallel"
    )
    
    parser.add_argument(
        "--list-sors",
        action="store_true",
        help="List all available SOR types and exit"
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save detailed JSON results (default: True)"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager()
    setup_logging(config_manager)
    processor = UnifiedSORProcessor(config_manager)
    
    # Handle list-sors mode
    if args.list_sors:
        enabled_sors = processor.get_enabled_sor_types()
        print("Available SOR types:")
        for sor_type in enabled_sors:
            config = processor.sor_config.get("sor_analysis", {}).get("sor_types", {}).get(sor_type, {})
            print(f"  {sor_type}: {config.get('description', 'No description')}")
        return
    
    # Validate required arguments
    if not args.parent_folder:
        print("Error: --parent-folder is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    try:
        # Determine SOR types to analyze
        sor_types = None
        if args.sor_list:
            sor_types = [s.strip() for s in args.sor_list.split(",")]
            sor_types = processor.validate_sor_types(sor_types)
            if not sor_types:
                print("Error: No valid SOR types specified")
                sys.exit(1)
        
        # Determine output path
        if args.output_path:
            output_base_path = args.output_path
        else:
            # Use default from config with timestamp
            default_output = processor.sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_output_folder", "artefacts/batch_results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base_path = f"{default_output}_{timestamp}"
        
        # Process work orders
        print(f"Starting unified SOR processing...")
        print(f"Parent folder: {args.parent_folder}")
        print(f"SOR types: {sor_types if sor_types else 'All enabled'}")
        print(f"Processing mode: {'Sequential' if args.sequential else 'Parallel'}")
        print(f"Output path: {output_base_path}")
        print("-" * 60)
        
        batch_results = await processor.process_work_orders_batch(
            args.parent_folder,
            sor_types=sor_types,
            sequential=args.sequential
        )
        
        # Save results
        saved_files = processor.save_results(batch_results, output_base_path)
        
        # Print summary
        summary = batch_results["batch_summary"]
        print("\n" + "="*60)
        print("UNIFIED SOR PROCESSING SUMMARY")
        print("="*60)
        print(f"Total work orders processed: {summary['total_work_orders']}")
        print(f"Successful work orders: {summary['successful_work_orders']}")
        print(f"Failed work orders: {summary['failed_work_orders']}")
        print(f"Total images processed: {summary['total_images']}")
        print(f"Total cost: ${summary['total_cost']:.4f}")
        print(f"Total tokens: {summary['total_input_tokens']} input, {summary['total_output_tokens']} output")
        
        if 'sor_statistics' in summary:
            print("\nSOR Results Summary:")
            for sor_type, stats in summary['sor_statistics'].items():
                print(f"  {sor_type}: {stats['pass']} PASS, {stats['fail']} FAIL, {stats['error']} ERROR")
        
        print(f"\nResults saved to: {output_base_path}")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Unified SOR processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 