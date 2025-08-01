from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from ..config import ConfigManager


class WorkOrderProcessor:
    """Processes multiple work order folders for batch SOR analysis."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.batch_config = self._get_batch_config()
    
    def _get_batch_config(self) -> Dict[str, Any]:
        """Get batch processing configuration from SOR analysis config."""
        sor_config = self.config_manager.get_config("sor_analysis_config")
        return sor_config.get("sor_analysis", {}).get("batch_processing", {
            "work_order_pattern": r"(\d+)",  # Extract numbers from folder names
            "exclude_patterns": ["__pycache__", ".git", "test_grids", "sor_results"],
            "min_images_per_work_order": 1,
            "max_work_orders": None,  # None = no limit
            "parallel_processing": True,
            "max_concurrent_work_orders": 3
        })
    
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
        pattern = self.batch_config["work_order_pattern"]
        exclude_patterns = self.batch_config["exclude_patterns"]
        min_images = self.batch_config["min_images_per_work_order"]
        
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
        
        # Apply max work orders limit
        max_work_orders = self.batch_config["max_work_orders"]
        if max_work_orders and len(work_orders) > max_work_orders:
            work_orders = work_orders[:max_work_orders]
            self.logger.info(f"Limited to {max_work_orders} work orders")
        
        self.logger.info(f"Discovered {len(work_orders)} work orders for processing")
        return work_orders
    
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """Count supported image files in a folder."""
        supported_formats = self.config_manager.get_image_processing_config().get("supported_formats", ["*.png", "*.jpg", "*.jpeg"])
        
        count = 0
        for format_pattern in supported_formats:
            count += len(list(folder_path.glob(format_pattern)))
        
        return count
    
    async def process_work_orders_batch(self, parent_folder: str, analysis_callback) -> Dict[str, Any]:
        """
        Process multiple work orders in batch.
        
        Args:
            parent_folder: Path to parent folder containing work order sub-folders
            analysis_callback: Async function to call for each work order analysis
            
        Returns:
            Dict with batch results and summary
        """
        work_orders = self.discover_work_orders(parent_folder)
        
        if not work_orders:
            raise ValueError(f"No valid work orders found in: {parent_folder}")
        
        # Process work orders
        if self.batch_config["parallel_processing"]:
            results = await self._process_work_orders_parallel(work_orders, analysis_callback)
        else:
            results = await self._process_work_orders_sequential(work_orders, analysis_callback)
        
        # Create batch summary
        summary = self._create_batch_summary(work_orders, results)
        
        return {
            "batch_summary": summary,
            "work_order_results": results
        }
    
    async def _process_work_orders_parallel(self, work_orders: List[Dict[str, Any]], analysis_callback) -> Dict[str, Any]:
        """Process work orders in parallel with concurrency control."""
        max_concurrent = self.batch_config["max_concurrent_work_orders"]
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_work_order(work_order: Dict[str, Any]) -> tuple[str, Any]:
            async with semaphore:
                try:
                    self.logger.info(f"Processing work order {work_order['work_order_number']}")
                    result = await analysis_callback(work_order)
                    return work_order["work_order_number"], result
                except Exception as e:
                    self.logger.error(f"Failed to process work order {work_order['work_order_number']}: {e}")
                    return work_order["work_order_number"], {"error": str(e)}
        
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
    
    async def _process_work_orders_sequential(self, work_orders: List[Dict[str, Any]], analysis_callback) -> Dict[str, Any]:
        """Process work orders sequentially."""
        results = {}
        
        for work_order in work_orders:
            try:
                self.logger.info(f"Processing work order {work_order['work_order_number']}")
                result = await analysis_callback(work_order)
                results[work_order["work_order_number"]] = result
            except Exception as e:
                self.logger.error(f"Failed to process work order {work_order['work_order_number']}: {e}")
                results[work_order["work_order_number"]] = {"error": str(e)}
        
        return results
    
    def _create_batch_summary(self, work_orders: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for the batch processing."""
        total_work_orders = len(work_orders)
        successful_work_orders = len([r for r in results.values() if "error" not in r])
        failed_work_orders = total_work_orders - successful_work_orders
        
        total_images = sum(wo["image_count"] for wo in work_orders)
        
        return {
            "total_work_orders": total_work_orders,
            "successful_work_orders": successful_work_orders,
            "failed_work_orders": failed_work_orders,
            "total_images": total_images,
            "processing_mode": "parallel" if self.batch_config["parallel_processing"] else "sequential",
            "max_concurrent": self.batch_config["max_concurrent_work_orders"]
        } 