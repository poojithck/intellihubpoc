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
        
        # Determine prompts subdirectory and version (allows switching to Targeted-Prompts/vX)
        self.prompts_subdir = self.sor_config.get("sor_analysis", {}).get("prompt_subdir", None)
        self.prompts_version = self.sor_config.get("sor_analysis", {}).get("prompt_version", None)

        # Pre-load all prompt configs to avoid concurrent loading during processing
        self.prompt_configs = {}
        self.model_params = config_manager.get_model_params(config_type=self.default_settings.get("model_config", "analysis"))
        
        for sor_type in self.get_enabled_sor_types():
            self.prompt_configs[sor_type] = config_manager.get_prompt_config(
                sor_type,
                prompts_subdir=self.prompts_subdir,
                prompts_version=self.prompts_version,
            )
        self.logger.info(f"Prompt directory: {self.prompts_subdir}{'/' + self.prompts_version if self.prompts_version else ''}")
        
        self.logger.info(f"Pre-loaded {len(self.prompt_configs)} prompt configurations")

        # Configure targeted feeder behavior
        targeted_cfg = self.sor_config.get("sor_analysis", {}).get("targeted_images", {})
        self.targeted_mode = targeted_cfg.get("enabled", False)
        self.folder_mapping = targeted_cfg.get("folder_mapping", {})
        self.default_folder_name = targeted_cfg.get("default_folder", "Meter Board")
        self.fallback_to_default_folder = bool(targeted_cfg.get("fallback_to_default_folder", True))
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
        """Count image files in a folder recursively (includes subfolders)."""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        return sum(1 for p in folder_path.rglob('*') if p.is_file() and p.suffix.lower() in supported_extensions)
    

    

    
   # Update the _analyze_sor_type method in UnifiedSORProcessor class

    async def _analyze_sor_type(self, sor_type: str, work_order_path: str, 
                           images: List[Dict[str, str]], work_order_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single SOR type using provided images (targeted or grids)."""
        
        # Get pre-loaded prompt configuration for this SOR type
        prompt_config = self.prompt_configs[sor_type]
        model_params = self.model_params

        # Initialize RAG pipeline if needed for supported SOR types
        if sor_type in ["MeterConsolidationE4", "FuseReplacement"]:
            try:
                from src.rag.pipeline import RAGPipeline
                rag_pipeline = RAGPipeline(self.config_manager)
                
                # Enhance prompts with RAG - now returns ONLY enhanced prompt config
                # Work order images are NOT modified
                prompt_config = rag_pipeline.enhance_sor_analysis(
                    sor_type=sor_type,
                    prompt_config=prompt_config,
                    work_order_images=images  # Passed for context only
                )
                
                self.logger.info(f"RAG enhancement applied for {sor_type}")
            except Exception as e:
                self.logger.warning(f"RAG enhancement failed, continuing without it: {e}")

        # Compose final prompt
        system_prompt = (prompt_config.get("system_prompt") or "").strip()
        main_prompt = (prompt_config.get("main_prompt") or "").strip()
        
        # Save the enhanced main prompt for debugging
        if prompt_config.get('rag_enhanced'):
            debug_dir = Path("debug_prompts")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"{sor_type}_{work_order_info['work_order_number']}_{timestamp}.txt"
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"SOR Type: {sor_type}\n")
                f.write(f"Work Order: {work_order_info['work_order_number']}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"RAG Enhanced: {prompt_config.get('rag_enhanced', False)}\n")
                f.write(f"Reference Count: {prompt_config.get('reference_count', 0)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("SYSTEM PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(system_prompt + "\n\n")
                f.write("=" * 80 + "\n\n")
                f.write("MAIN PROMPT (with reference images):\n")
                f.write("-" * 40 + "\n")
                f.write(main_prompt + "\n")
            
            self.logger.info(f"Saved debug prompt to: {debug_file}")
        
        # If no images, short-circuit with deterministic FAIL
        if not images:
            boolean_field = self.sor_config.get("sor_analysis", {}).get("table_generation", {}).get("boolean_fields", {}).get(sor_type)
            result_base = {
                "sor_type": sor_type,
                "work_order": work_order_info["work_order_number"],
                "folder_name": work_order_info["folder_name"],
                "analysis_timestamp": datetime.now().isoformat(),
                "notes": "No targeted evidence provided for this SOR",
                "Notes": "No targeted evidence provided for this SOR",
                "total_cost": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
            if boolean_field:
                result_base[boolean_field] = False
            return result_base

        # Run synchronous Bedrock call in thread pool for true async concurrency
        # Work order images are passed here - they are NOT mixed with reference images
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.bedrock_client.invoke_model_multi_image(
                    prompt=main_prompt,
                    system_prompt=system_prompt,
                    images=images,  # These are work order images only
                    max_tokens=model_params.get("max_tokens"),
                    temperature=model_params.get("temperature")
                )
            )
        
        # Parse response
        response_text = response.get("text", "")
        from src.clients.bedrock_client import BedrockClient
        response_text = BedrockClient.repair_json_response(response_text)
        
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
        
        def prepare_images_sync(work_order: Dict[str, Any]) -> tuple[str, Any]:
            """Prepare encoded images (targeted or grids) for thread pool execution."""
            work_order_number = work_order["work_order_number"]
            work_order_path = work_order["path"]
            try:
                if self.targeted_mode:
                    from src.tools.image_loader import ImageLoader
                    encoded_by_sor: Dict[str, list] = {}
                    folder_cache: Dict[Path, list] = {}

                    def load_folder(folder_path: Path) -> list:
                        if folder_path in folder_cache:
                            return folder_cache[folder_path]
                        # Use AWS Bedrock's 5MB limit
                        loader = ImageLoader(str(folder_path), max_size_mb=5.0)
                        images = loader.load_images_to_memory(single=False)
                        if not images:
                            return []
                        # Targeted mode: prefer JPEG for photo content to reduce size and speed up
                        encoded = loader.encode_images(images, format="JPEG")
                        # Inject media_type for Bedrock
                        for item in encoded:
                            item["media_type"] = "image/jpeg"
                        folder_cache[folder_path] = encoded
                        return encoded

                    wo_path = Path(work_order_path)
                    for sor_type in self.get_enabled_sor_types():
                        folder_names = self.folder_mapping.get(sor_type, [])
                        selected_imgs: list = []
                        for fname in folder_names:
                            candidate = wo_path / fname
                            if candidate.exists() and candidate.is_dir():
                                imgs = load_folder(candidate)
                                if imgs:
                                    selected_imgs.extend(imgs)
                        # Fallback to default folder if enabled and nothing found
                        if not selected_imgs and self.fallback_to_default_folder:
                            fallback = wo_path / self.default_folder_name
                            if fallback.exists() and fallback.is_dir():
                                selected_imgs = load_folder(fallback)
                        # Optionally cap images per SOR for performance (keep chronological order)
                        if self.max_images_per_sor and len(selected_imgs) > self.max_images_per_sor:
                            selected_imgs = selected_imgs[:self.max_images_per_sor]
                        encoded_by_sor[sor_type] = selected_imgs or []

                    return work_order_number, (work_order, encoded_by_sor)
                else:
                    # Legacy grid path
                    self.logger.info(f"Creating grids for work order {work_order_number}")
                    grids = self.gridder.create_grids(work_order_path, output_dir=None)
                    if not grids:
                        raise RuntimeError(f"No grid images could be created for work order {work_order_number}")
                    output_format = self.default_settings.get("output_format", "PNG")
                    # Use AWS Bedrock's 5MB limit for grid images
                    encoded_grids = self.gridder.encode_grids(grids, format=output_format, max_size_mb=5.0)
                    del grids
                    return work_order_number, (work_order, {"__grids__": encoded_grids})
            except Exception as e:
                self.logger.error(f"Failed to prepare images for work order {work_order_number}: {e}")
                return work_order_number, e

        async def process_work_order_parallel(work_order: Dict[str, Any]) -> tuple[str, Any]:
            try:
                work_order_number = work_order["work_order_number"]
                self.logger.info(f"Processing work order {work_order_number}")
                
                # Run grid creation in thread pool for true parallelism
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    prepared_number, prepared_result = await loop.run_in_executor(
                        executor, prepare_images_sync, work_order
                    )
                
                if isinstance(prepared_result, Exception):
                    raise prepared_result
                
                work_order, prepared_images = prepared_result
                
                # Process SOR types without work order level semaphore
                # Individual SOR processing will be controlled by global semaphore
                result = await self._process_sor_types_for_work_order(
                    work_order, sor_types, prepared_images
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
                                               prepared_images: Any) -> Dict[str, Any]:
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
                        
                        # Choose images per mode
                        if self.targeted_mode:
                            images = prepared_images.get(sor_type, [])
                        else:
                            images = prepared_images.get("__grids__", [])

                        result = await self._analyze_sor_type(sor_type, work_order_path, images, work_order)
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
    
    def _get_batch_metadata(self) -> Dict[str, Any]:
        """Get comprehensive batch metadata including prompt and model configuration."""
        from datetime import datetime
        
        # Get AWS/Bedrock configuration
        aws_config = self.config_manager.get_aws_config()
        
        # Build prompt path
        prompt_path = self.prompts_subdir if self.prompts_subdir else "prompts"
        if self.prompts_version:
            prompt_path += f"/{self.prompts_version}"
        
        return {
            "client_type": "aws_bedrock",
            "model_configuration": {
                "model_id": aws_config.get("bedrock", {}).get("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                "region": aws_config.get("aws", {}).get("region", "us-east-1"),
                "timeout": aws_config.get("bedrock", {}).get("timeout", 30),
                "retry_attempts": aws_config.get("bedrock", {}).get("retry_attempts", 3)
            },
            "prompt_configuration": {
                "prompt_path": prompt_path,
                "prompt_subdir": self.prompts_subdir,
                "prompt_version": self.prompts_version,
                "enabled_sor_types": self.get_enabled_sor_types()
            },
            "processing_configuration": {
                "targeted_mode": self.targeted_mode,
                "max_images_per_sor": self.max_images_per_sor,
                "max_concurrent_work_orders": self.batch_config.get("max_concurrent_work_orders", 8),
                "max_concurrent_sor_types": self.batch_config.get("max_concurrent_sor_types", 8),
                "batch_size": self.batch_config.get("batch_size", 5)
            },
            "metadata_generated_at": datetime.now().isoformat()
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
            confidence_scores = []
            
            # Get the correct boolean field name for this SOR type
            boolean_field = boolean_field_config.get(sor_type)
            
            for work_order_result in results.values():
                sor_result = work_order_result.get("sor_results", {}).get(sor_type, {})
                if "error" in sor_result:
                    error_count += 1
                else:
                    # Collect confidence score
                    if "confidence_score" in sor_result:
                        confidence_scores.append(sor_result["confidence_score"])
                    
                    # Check pass/fail
                    if boolean_field and sor_result.get(boolean_field):
                        boolean_value = sor_result.get(boolean_field)
                        if isinstance(boolean_value, bool):
                            if boolean_value:
                                pass_count += 1
                            else:
                                fail_count += 1
                        elif isinstance(boolean_value, str):
                            if boolean_value.upper() in ["PASS", "TRUE", "YES", "1"]:
                                pass_count += 1
                            else:
                                fail_count += 1
                        else:
                            fail_count += 1
                    else:
                        fail_count += 1
            
            # Calculate confidence statistics
            confidence_stats = {}
            if confidence_scores:
                confidence_stats = {
                    "average": sum(confidence_scores) / len(confidence_scores),
                    "min": min(confidence_scores),
                    "max": max(confidence_scores),
                    "high_confidence": sum(1 for s in confidence_scores if s >= 80),
                    "medium_confidence": sum(1 for s in confidence_scores if 60 <= s < 80),
                    "low_confidence": sum(1 for s in confidence_scores if s < 60)
                }
            
            sor_statistics[sor_type] = {
                "pass": pass_count,
                "fail": fail_count,
                "error": error_count,
                "confidence_stats": confidence_stats
            }
        
        # Add batch configuration metadata
        batch_metadata = self._get_batch_metadata()
        
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
            "api_rate_limit_delay": self.batch_config.get("api_rate_limit_delay", 0.05),
            "batch_metadata": batch_metadata
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> Dict[str, str]:
        """Save results in minimal required formats: CSV table and summary.json."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Generate and save table (CSV only)
        try:
            df = self.table_generator.generate_table_from_batch_results(results)
            if not df.empty:
                # Save CSV
                csv_file = self.table_generator.save_table(df, str(output_dir / "sor_results"), "csv")
                saved_files["csv"] = csv_file
        except Exception as e:
            self.logger.error(f"Failed to generate CSV table: {e}")
        
        # Save batch summary as summary.json
        try:
            summary = results.get("batch_summary", {})
            summary_file = output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            saved_files["summary"] = str(summary_file)
        except Exception as e:
            self.logger.error(f"Failed to save summary.json: {e}")
        
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
    
    # Attach file handler for batch logs
    try:
        output_dir = Path(config["output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.getLogger().level)
        from src.utils.cli import setup_logging as _setup_logging
        formatter = logging.Formatter(config_manager.get_logging_config().get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to attach batch file logger: {e}")
    
    # Apply prompt directory/version overrides from CLI if provided
    try:
        if config.get("prompt_subdir") or config.get("prompt_version"):
            if config.get("prompt_subdir"):
                processor.prompts_subdir = config.get("prompt_subdir")
            if config.get("prompt_version"):
                processor.prompts_version = config.get("prompt_version")
            # Rebuild prompt configs with overrides
            processor.prompt_configs = {}
            for sor_type in processor.get_enabled_sor_types():
                processor.prompt_configs[sor_type] = processor.config_manager.get_prompt_config(
                    sor_type,
                    prompts_subdir=processor.prompts_subdir,
                    prompts_version=processor.prompts_version,
                )
            logging.getLogger(__name__).info(
                f"Prompt directory override in effect: {processor.prompts_subdir}{'/' + processor.prompts_version if processor.prompts_version else ''}"
            )
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to apply prompt overrides: {e}")

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