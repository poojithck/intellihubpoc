#!/usr/bin/env python3
"""
IntelliHub SOR POC - Main Orchestrator

This script acts as the main entry point and orchestrator for the SOR analysis system.
It automatically selects the appropriate processor (AWS or Azure) based on configuration.
"""

import asyncio
import logging
from src.config import ConfigManager
from src.utils import setup_logging


async def main():
    """Main orchestration function."""
    try:
        # Initialize configuration manager first
        config_manager = ConfigManager()
        
        # Setup logging with config manager
        setup_logging(config_manager)
        logger = logging.getLogger(__name__)
        
        # Get SOR analysis configuration
        sor_config = config_manager.get_config("sor_analysis_config")
        client_type = sor_config.get("sor_analysis", {}).get("client_type", "aws")
        
        logger.info(f"Initializing SOR analysis with client type: {client_type}")
        
        if client_type.lower() == "azure":
            # Use Azure OpenAI processor
            from scripts.azure_unified_sor_processor import AzureUnifiedSORProcessor
            processor = AzureUnifiedSORProcessor(config_manager)
            logger.info("Using Azure OpenAI processor")
            
        elif client_type.lower() == "aws":
            # Use AWS Bedrock processor
            from scripts.unified_sor_processor import UnifiedSORProcessor
            processor = UnifiedSORProcessor(config_manager)
            logger.info("Using AWS Bedrock processor")
            
        else:
            raise ValueError(f"Unsupported client type: {client_type}. Must be 'aws' or 'azure'")
        
        # Get batch processing configuration
        batch_config = sor_config.get("sor_analysis", {}).get("batch_processing", {})
        # Get parent folder from default_paths (correct location)
        parent_folder = sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_parent_folder", "artefacts")
        max_work_orders = batch_config.get("max_work_orders", 10)
        batch_size = batch_config.get("batch_size", 5)
        
        logger.info(f"Starting batch processing with {client_type.upper()} client")
        logger.info(f"Parent folder: {parent_folder}")
        logger.info(f"Max work orders: {max_work_orders}")
        logger.info(f"Batch size: {batch_size}")
        
        # Process work orders
        results = await processor.process_work_orders_batch(
            parent_folder=parent_folder,
            max_work_orders=max_work_orders,
            batch_size=batch_size
        )
        
        # Generate results table using configured batch output folder
        batch_output_folder = sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_output_folder", "outputs/batch_results")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{batch_output_folder}_{timestamp}"
        table_path = processor.generate_results_table(results, output_path)
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Results table generated at: {table_path}")
        # Handle both Azure (batch_summary) and Bedrock (summary) return structures
        summary_key = 'batch_summary' if 'batch_summary' in results else 'summary'
        logger.info(f"Total work orders processed: {results[summary_key]['total_work_orders']}")
        logger.info(f"Total cost: ${results[summary_key]['total_cost']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to process SOR analysis: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
