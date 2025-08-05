#!/usr/bin/env python3
"""
Unified SOR Processor

A comprehensive system that efficiently processes multiple work orders and SOR types.
Key features:
- Discovers work orders in parent folder
- Generates single grid per work order (efficient)
- Processes all SOR types using shared grids
- Generates comprehensive output (tables, summaries, JSON)
- Enhanced parallelization with concurrent SOR processing

Usage:
    # Process all work orders with all SOR types (uses config default folder)
    python unified_sor_processor.py
    
    # Process with custom folder
    python unified_sor_processor.py --parent-folder path/to/work_orders
    
    # Process with specific SOR types only
    python unified_sor_processor.py --sor-list AsbestosBagAndBoard,FuseReplacement
    
    # Test mode: Process limited work orders (uses config default limit)
    python unified_sor_processor.py --test-mode
    
    # Process specific number of work orders
    python unified_sor_processor.py --max-work-orders 10
    
    # Process with custom output format
    python unified_sor_processor.py --output-format excel
    
    # Process with parallel processing disabled
    python unified_sor_processor.py --sequential
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
import time
import concurrent.futures

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.tools import ResultsTableGenerator
from src.tools.image_gridder import ImageGridder
from src.utils import setup_logging, CLIConfig


class UnifiedSORProcessor:
    """
    Unified processor for efficient SOR analysis across multiple work orders.
    
    Key optimizations:
    - Single grid generation per work order
    - Shared grid usage across all SOR types
    - Efficient batch processing
    - Comprehensive output generation
    - Enhanced parallelization with concurrent SOR processing
    - API call batching for improved throughput
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.sor_config = config_manager.get_config("sor_analysis_config")
        # Get work order discovery configuration
        self.work_order_config = self.sor_config.get("sor_analysis", {}).get("batch_processing", {})
        self.table_generator = ResultsTableGenerator(config_manager)
        self.gridder = ImageGridder(config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Create shared BedrockClient for efficiency
        from src.clients.bedrock_client import BedrockClient
        self.bedrock_client = BedrockClient.from_config(config_manager)
        
        # Get configuration
        self.batch_config = self.sor_config.get("sor_analysis", {}).get("batch_processing", {})
        self.default_settings = self.sor_config.get("sor_analysis", {}).get("default_settings", {})
        
        # Pre-load all prompt configs to avoid concurrent loading during processing
        self.prompt_configs = {}
        self.model_params = config_manager.get_model_params(config_type=self.default_settings.get("model_config", "analysis"))
        
        for sor_type in self.get_enabled_sor_types():
            self.prompt_configs[sor_type] = config_manager.get_prompt_config(sor_type)
        
        self.logger.info(f"Pre-loaded {len(self.prompt_configs)} prompt configurations")
        
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
    
    def discover_work_orders(self, parent_folder: str) -> List[Dict[str, Any]]:
        """
        Discover work order folders in the parent directory.
        
        Args:
            parent_folder: Path to parent folder containing work order sub-folders
            
        Returns:
            List of work order info dicts with 'path', 'work_order_number', 'image_count'
        """
        parent_path = Path(parent_folder)
        if not parent_path.exists() or not parent_path.is_dir():
            raise ValueError(f"Parent folder not found or not a directory: {parent_folder}")
        
        work_orders = []
        exclude_patterns = self.work_order_config.get("exclude_patterns", [])
        min_images = self.work_order_config.get("min_images_per_work_order", 1)
        
        for subfolder in parent_path.iterdir():
            if not subfolder.is_dir():
                continue
            
            # Skip excluded patterns
            if any(exclude in subfolder.name for exclude in exclude_patterns):
                continue
            
            # Use folder name as work order number
            work_order_number = subfolder.name
            
            # Count images in the work order folder
            image_count = self._count_images_in_folder(subfolder)
            
            if image_count >= min_images:
                work_orders.append({
                    "path": str(subfolder),
                    "work_order_number": work_order_number,
                    "folder_name": subfolder.name,
                    "image_count": image_count
                })
                self.logger.info(f"Found work order {work_order_number} with {image_count} images")
            else:
                self.logger.warning(f"Work order {work_order_number} has insufficient images ({image_count})")
        
        # Sort by work order number (alphabetical)
        work_orders.sort(key=lambda x: x["work_order_number"])
        
        self.logger.info(f"Discovered {len(work_orders)} work orders for processing")
        return work_orders
    
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """Count image files in a folder."""
        from src.tools.image_loader import ImageLoader
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        
        count = 0
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                count += 1
        
        return count
    

    

    
    async def _analyze_sor_type(self, sor_type: str, work_order_path: str, 
                               encoded_grids: List[Dict[str, str]], work_order_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single SOR type using pre-encoded grids and shared BedrockClient."""
        

        
        # Get pre-loaded prompt configuration for this SOR type
        prompt_config = self.prompt_configs[sor_type]
        model_params = self.model_params
        
        # Run synchronous Bedrock call in thread pool for true async concurrency
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.bedrock_client.invoke_model_multi_image(
                    prompt=prompt_config["main_prompt"],
                    images=encoded_grids,
                    max_tokens=model_params.get("max_tokens"),
                    temperature=model_params.get("temperature")
                )
            )
        
        # Parse response using shared client
        response_text = response.get("text", "")
        from src.clients.bedrock_client import BedrockClient
        response_text = BedrockClient.repair_json_response(response_text)
        
        # Create simple fallback parser for this SOR type
        fallback_parser = BedrockClient.create_fallback_parser(sor_type)
        
        parsed_response = self.bedrock_client.parse_json_response(
            response_text,
            fallback_parser
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
                                      max_work_orders: Optional[int] = None,
                                      batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multiple work orders in memory-efficient batches with parallel grid generation.
        
        Args:
            parent_folder: Path to parent folder containing work order sub-folders
            sor_types: List of SOR types to analyze (None = all enabled)
            max_work_orders: Maximum number of work orders to process (None = all)
            batch_size: Number of work orders to process per memory batch (None = use config default)
            
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
        work_orders = self.discover_work_orders(parent_folder)
        
        if not work_orders:
            raise ValueError(f"No valid work orders found in: {parent_folder}")
        
        # Apply work order limit if specified
        if max_work_orders and max_work_orders > 0:
            original_count = len(work_orders)
            work_orders = work_orders[:max_work_orders]
            self.logger.info(f"Limited to {len(work_orders)} work orders (from {original_count} total)")
        
        # Get batch size from config if not provided
        if batch_size is None:
            batch_size = self.batch_config.get("batch_size", 10)
        
        self.logger.info(f"Found {len(work_orders)} work orders to process in batches of {batch_size}")
        
        # Process work orders in memory-efficient batches
        overall_results = {}
        
        for batch_idx in range(0, len(work_orders), batch_size):
            batch_end = min(batch_idx + batch_size, len(work_orders))
            current_batch = work_orders[batch_idx:batch_end]
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(work_orders) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(current_batch)} work orders)")
            
            # Process current batch
            batch_results = await self._process_work_orders_parallel(current_batch, sor_types)
            
            # Merge results into overall results
            overall_results.update(batch_results)
            
            # Force garbage collection after each batch to free memory
            import gc
            gc.collect()
            
            self.logger.info(f"Completed batch {batch_num}/{total_batches}, total results: {len(overall_results)}")
        
        # Create batch summary using all work orders and results
        batch_summary = self._create_batch_summary(work_orders, overall_results, sor_types)
        
        return {
            "batch_summary": batch_summary,
            "work_order_results": overall_results
        }
    

    
    async def _process_work_orders_parallel(self, work_orders: List[Dict[str, Any]], 
                                          sor_types: List[str]) -> Dict[str, Any]:
        """Process work orders in parallel with concurrency control."""
        # Create global semaphore for individual SOR processing across all work orders
        max_concurrent_sors = self.batch_config.get("max_concurrent_work_orders", 8) * self.batch_config.get("max_concurrent_sor_types", 8)
        self.global_sor_semaphore = asyncio.Semaphore(max_concurrent_sors)
        
        def create_grids_sync(work_order: Dict[str, Any]) -> tuple[str, Any]:
            """Synchronous grid creation function for thread pool execution."""
            work_order_number = work_order["work_order_number"]
            work_order_path = work_order["path"]
            
            try:
                self.logger.info(f"Creating grids for work order {work_order_number}")
                
                # Generate grid images for this work order (CPU intensive, memory-only)
                grids = self.gridder.create_grids(work_order_path, output_dir=None)
                
                if not grids:
                    raise RuntimeError(f"No grid images could be created for work order {work_order_number}")
                
                # Encode grids (this will also clean up PIL images)
                output_format = self.default_settings.get("output_format", "PNG")
                encoded_grids = self.gridder.encode_grids(grids, format=output_format)
                
                self.logger.info(f"Created {len(encoded_grids)} grid images for work order {work_order_number}")
                
                # Additional cleanup: explicitly delete grids list to free memory
                del grids
                
                return work_order_number, (work_order, encoded_grids)
                
            except Exception as e:
                self.logger.error(f"Failed to create grids for work order {work_order_number}: {e}")
                return work_order_number, e

        async def process_work_order_parallel(work_order: Dict[str, Any]) -> tuple[str, Any]:
            try:
                work_order_number = work_order["work_order_number"]
                self.logger.info(f"Processing work order {work_order_number}")
                
                # Run grid creation in thread pool for true parallelism
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    grid_result_number, grid_result = await loop.run_in_executor(
                        executor, create_grids_sync, work_order
                    )
                
                if isinstance(grid_result, Exception):
                    raise grid_result
                
                work_order, encoded_grids = grid_result
                
                # Process SOR types without work order level semaphore
                # Individual SOR processing will be controlled by global semaphore
                result = await self._process_sor_types_for_work_order(
                    work_order, sor_types, encoded_grids
                )
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
        tasks = [process_work_order_parallel(wo) for wo in work_orders]
        
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
    
    async def _process_sor_types_for_work_order(self, work_order: Dict[str, Any], 
                                              sor_types: List[str], 
                                              encoded_grids: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process SOR types for a single work order with pre-created grids."""
        work_order_number = work_order["work_order_number"]
        work_order_path = work_order["path"]
        
        # Process SOR types using shared encoded grids - with concurrent processing
        api_delay = self.batch_config.get("api_rate_limit_delay")
        
        results = {}
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        async def process_single_sor_type(sor_type: str) -> tuple[str, Any]:
            """Process a single SOR type with concurrency control and rate limiting."""
            # Use global semaphore to control concurrent SOR processing across ALL work orders
            async with self.global_sor_semaphore:
                max_retries = 10  # Maximum number of retries for throttling
                base_delay = 1.0  # Base delay in seconds
                
                for attempt in range(max_retries + 1):
                    try:
                        # Progressive delay for rate limiting and retries
                        if attempt > 0:
                            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s, 60s, 60s
                            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
                            self.logger.info(f"Retrying {sor_type} for work order {work_order_number} (attempt {attempt + 1}/{max_retries + 1}) after {delay}s delay")
                            await asyncio.sleep(delay)
                        elif api_delay > 0:
                            # Small delay to prevent API rate limiting on first attempt
                            await asyncio.sleep(api_delay)
                        
                        result = await self._analyze_sor_type(sor_type, work_order_path, encoded_grids, work_order)
                        self.logger.info(f"Completed {sor_type} analysis for work order {work_order_number}")
                        return sor_type, result
                        
                    except Exception as e:
                        error_message = str(e)
                        is_retryable = (
                            "ThrottlingException" in error_message or
                            "TooManyRequestsException" in error_message or
                            "ServiceUnavailableException" in error_message or
                            "InternalServerError" in error_message or
                            "Timeout" in error_message
                        )
                        
                        if is_retryable and attempt < max_retries:
                            self.logger.warning(f"Retryable error for {sor_type} (work order {work_order_number}): {error_message}")
                            continue  # Retry
                        else:
                            # Final failure or non-retryable error
                            self.logger.error(f"Failed to analyze {sor_type} for work order {work_order_number} after {attempt + 1} attempts: {e}")
                            error_result = {
                                "sor_type": sor_type,
                                "work_order": work_order_number,
                                "error": str(e),
                                "total_cost": 0,
                                "input_tokens": 0,
                                "output_tokens": 0
                            }
                            return sor_type, error_result
        
        # Create concurrent tasks for all SOR types
        sor_tasks = [process_single_sor_type(sor_type) for sor_type in sor_types]
        
        # Execute all SOR analysis tasks concurrently
        sor_results_list = await asyncio.gather(*sor_tasks, return_exceptions=True)
        
        # Process results and accumulate costs
        for result in sor_results_list:
            if isinstance(result, Exception):
                self.logger.error(f"SOR task failed with exception: {result}")
                continue
                
            sor_type, sor_result = result
            results[sor_type] = sor_result
            
            # Accumulate costs safely
            total_cost += sor_result.get("total_cost", 0)
            total_input_tokens += sor_result.get("input_tokens", 0)
            total_output_tokens += sor_result.get("output_tokens", 0)
        
        # Create work order summary
        summary = {
            "work_order": work_order_number,
            "folder_name": work_order["folder_name"],
            "image_count": work_order["image_count"],
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
        
        # SOR-specific statistics using correct boolean field names from config
        sor_statistics = {}
        boolean_field_config = self.sor_config.get("sor_analysis", {}).get("table_generation", {}).get("boolean_fields", {})
        
        for sor_type in sor_types:
            pass_count = 0
            fail_count = 0
            error_count = 0
            
            # Get the correct boolean field name for this SOR type
            boolean_field = boolean_field_config.get(sor_type)
            
            for work_order_result in results.values():
                sor_result = work_order_result.get("sor_results", {}).get(sor_type, {})
                if "error" in sor_result:
                    error_count += 1
                elif boolean_field and sor_result.get(boolean_field):
                    # Check if the boolean field exists and is truthy
                    boolean_value = sor_result.get(boolean_field)
                    if isinstance(boolean_value, bool):
                        if boolean_value:
                            pass_count += 1
                        else:
                            fail_count += 1
                    elif isinstance(boolean_value, str):
                        # Handle string boolean values like "PASS", "FAIL", "True", "False"
                        if boolean_value.upper() in ["PASS", "TRUE", "YES", "1"]:
                            pass_count += 1
                        else:
                            fail_count += 1
                    else:
                        # If boolean field exists but isn't recognizable, count as fail
                        fail_count += 1
                else:
                    # If no boolean field configured or field not found, count as fail
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
            "processing_mode": "parallel_sors_across_work_orders",
            "max_concurrent_sors_global": self.batch_config.get("max_concurrent_work_orders", 8) * self.batch_config.get("max_concurrent_sor_types", 8),
            "concurrent_work_orders": "unlimited",
            "api_rate_limit_delay": self.batch_config.get("api_rate_limit_delay", 0.05)
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
    """Main function with CLI configuration and processing."""
    # Initialize configuration
    config_manager = ConfigManager()
    setup_logging(config_manager)
    
    # Set up CLI configuration
    cli_config = CLIConfig(config_manager)
    parser = cli_config.create_parser()
    args = parser.parse_args()
    
    # Initialize processor
    processor = UnifiedSORProcessor(config_manager)
    
    # Resolve configuration from CLI args and config defaults
    config = cli_config.resolve_config(args)
    
    # Handle list-sors mode
    if config["list_sors"]:
        enabled_sors = processor.get_enabled_sor_types()
        print("Available SOR types:")
        for sor_type in enabled_sors:
            sor_config = processor.sor_config.get("sor_analysis", {}).get("sor_types", {}).get(sor_type, {})
            print(f"  {sor_type}: {sor_config.get('description', 'No description')}")
        return
    
    try:
        # Validate SOR types if specified
        if config["sor_types"]:
            config["sor_types"] = processor.validate_sor_types(config["sor_types"])
            if not config["sor_types"]:
                print("Error: No valid SOR types specified")
                sys.exit(1)
        
        # Print processing information
        cli_config.print_processing_info(config)
        
        start_time = time.time()
        
        batch_results = await processor.process_work_orders_batch(
            config["parent_folder"],
            sor_types=config["sor_types"],
            max_work_orders=config["max_work_orders"],
            batch_size=config["batch_size"]
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save results
        saved_files = processor.save_results(batch_results, config["output_path"])
        
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
        print(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        print(f"Average time per work order: {processing_time/summary['total_work_orders']:.2f} seconds")
        
        if 'sor_statistics' in summary:
            print("\nSOR Results Summary:")
            for sor_type, stats in summary['sor_statistics'].items():
                print(f"  {sor_type}: {stats['pass']} PASS, {stats['fail']} FAIL, {stats['error']} ERROR")
        
        print(f"\nResults saved to: {config['output_path']}")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Unified SOR processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 