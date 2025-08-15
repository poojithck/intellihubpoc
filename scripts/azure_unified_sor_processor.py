from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import ConfigManager
from src.tools.results_table_generator import ResultsTableGenerator
from src.tools.image_gridder import ImageGridder
from src.utils import setup_logging, CLIConfig


class AzureUnifiedSORProcessor:
    """
    Azure OpenAI-based unified processor for efficient SOR analysis across multiple work orders.
    
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
        
        # Create shared Azure OpenAI client for efficiency
        from src.clients.azure_openai_client import AzureOpenAIClient
        self.azure_client = AzureOpenAIClient.from_config(config_manager)
        
        # Get configuration
        self.batch_config = self.sor_config.get("sor_analysis", {}).get("batch_processing", {})
        self.default_settings = self.sor_config.get("sor_analysis", {}).get("default_settings", {})
        
        # Pre-load all prompt configs to avoid concurrent loading during processing
        self.prompt_configs = {}
        self.model_params = config_manager.get_model_params(config_type=self.default_settings.get("model_config", "analysis"))
        
        for sor_type in self.get_enabled_sor_types():
            self.prompt_configs[sor_type] = config_manager.get_prompt_config(sor_type)
        
        self.logger.info(f"Pre-loaded {len(self.prompt_configs)} prompt configurations")
        
        # Configure targeted feeder behavior
        targeted_cfg = self.sor_config.get("sor_analysis", {}).get("targeted_images", {})
        self.targeted_mode = targeted_cfg.get("enabled", False)
        self.folder_mapping = targeted_cfg.get("folder_mapping", {})
        self.default_folder_name = targeted_cfg.get("default_folder", "Meter Board")
        self.max_images_per_sor = int(targeted_cfg.get("max_images_per_sor", 0) or 0)
        
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
        """Analyze a single SOR type using pre-encoded grids and shared Azure OpenAI client."""
        

        
        # Get pre-loaded prompt configuration for this SOR type
        prompt_config = self.prompt_configs[sor_type]
        model_params = self.model_params
        
        # Compose final prompt: include system + main for clarity (system may contain strict definitions)
        system_prompt = (prompt_config.get("system_prompt") or "").strip()
        main_prompt = (prompt_config.get("main_prompt") or "").strip()
        prompt_text = (f"{system_prompt}\n\n---\n\n{main_prompt}" if system_prompt else main_prompt).strip()
        
        # Debug preview to verify correct prompt dispatch per SOR
        try:
            preview = (prompt_text[:300] + ("…" if len(prompt_text) > 300 else ""))
            self.logger.debug(f"Prompt preview for {sor_type}: {preview}")
        except Exception:
            pass
        
        # Run synchronous Azure OpenAI call in thread pool for true async concurrency
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.azure_client.invoke_model_multi_image(
                    prompt=prompt_text,
                    images=encoded_grids,
                    max_tokens=model_params.get("max_tokens"),
                    temperature=model_params.get("temperature")
                )
            )
        
        # Parse response using shared client
        response_text = response.get("text", "")
        from src.clients.azure_openai_client import AzureOpenAIClient
        response_text = AzureOpenAIClient.repair_json_response(response_text)
        
        # Create simple fallback parser for this SOR type
        fallback_parser = AzureOpenAIClient.create_fallback_parser(sor_type)
        
        parsed_response = self.azure_client.parse_json_response(
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
        
        # Generate summary
        summary = self._generate_batch_summary(overall_results)
        
        return {
            "summary": summary,
            "results": overall_results
        }
    
    async def _process_work_orders_parallel(self, work_orders: List[Dict[str, Any]], 
                                          sor_types: List[str]) -> Dict[str, Any]:
        """Process multiple work orders in parallel with shared grid generation."""
        results = {}
        
        # Process work orders concurrently
        tasks = []
        for work_order in work_orders:
            task = self._process_single_work_order(work_order, sor_types)
            tasks.append(task)
        
        # Execute all tasks concurrently
        work_order_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(work_order_results):
            work_order = work_orders[i]
            work_order_number = work_order["work_order_number"]
            
            if isinstance(result, Exception):
                self.logger.error(f"Work order {work_order_number} failed: {result}")
                results[work_order_number] = {
                    "summary": {
                        "work_order": work_order_number,
                        "error": str(result)
                    },
                    "sor_results": {}
                }
            else:
                results[work_order_number] = result
        
        return results
    
    async def _process_single_work_order(self, work_order: Dict[str, Any], 
                                       sor_types: List[str]) -> Dict[str, Any]:
        """Process a single work order with all SOR types."""
        work_order_number = work_order["work_order_number"]
        work_order_path = work_order["path"]
        
        self.logger.info(f"Processing work order {work_order_number}")
        
        # Create grids for this work order
        grids = self.gridder.create_grids(work_order_path, output_dir=None)
        if not grids:
            raise RuntimeError(f"No grid images could be created for work order {work_order_number}")
        
        # Encode grids
        output_format = self.default_settings.get("output_format", "PNG")
        encoded_grids = self.gridder.encode_grids(grids, format=output_format)
        
        # Clean up grid objects to free memory
        del grids
        
        # Process all SOR types for this work order
        sor_results = {}
        
        # Process SOR types concurrently
        tasks = []
        for sor_type in sor_types:
            task = self._analyze_sor_type(sor_type, work_order_path, encoded_grids, work_order)
            tasks.append(task)
        
        # Execute all SOR analysis tasks concurrently
        sor_analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process SOR results
        for i, result in enumerate(sor_analysis_results):
            sor_type = sor_types[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"SOR {sor_type} failed for work order {work_order_number}: {result}")
                sor_results[sor_type] = {
                    "error": str(result),
                    "status": "error"
                }
            else:
                sor_results[sor_type] = result
        
        # Clean up encoded grids to free memory
        del encoded_grids
        
        # Create work order summary
        summary = {
            "work_order": work_order_number,
            "folder_name": work_order["folder_name"],
            "image_count": work_order["image_count"],
            "sor_types_analyzed": list(sor_results.keys()),
            "analysis_timestamp": datetime.now().isoformat(),
            "total_cost": sum(result.get("total_cost", 0) for result in sor_results.values() if isinstance(result, dict)),
            "input_tokens": sum(result.get("input_tokens", 0) for result in sor_results.values() if isinstance(result, dict)),
            "output_tokens": sum(result.get("output_tokens", 0) for result in sor_results.values() if isinstance(result, dict))
        }
        
        return {
            "summary": summary,
            "sor_results": sor_results
        }
    
    def _generate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the entire batch."""
        total_work_orders = len(results)
        successful_work_orders = sum(1 for r in results.values() if "error" not in r.get("summary", {}))
        failed_work_orders = total_work_orders - successful_work_orders
        
        total_cost = sum(r.get("summary", {}).get("total_cost", 0) for r in results.values())
        total_input_tokens = sum(r.get("summary", {}).get("input_tokens", 0) for r in results.values())
        total_output_tokens = sum(r.get("summary", {}).get("output_tokens", 0) for r in results.values())
        
        return {
            "total_work_orders": total_work_orders,
            "successful_work_orders": successful_work_orders,
            "failed_work_orders": failed_work_orders,
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "analysis_timestamp": datetime.now().isoformat(),
            "client_type": "azure_openai"
        }
    
    def generate_results_table(self, batch_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate results table from batch results."""
        return self.table_generator.generate_table_from_batch_results(batch_results, output_path)
