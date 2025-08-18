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

        # Resolve output folder for this batch early, and attach file logger
        batch_output_folder = sor_config.get("sor_analysis", {}).get("default_paths", {}).get("batch_output_folder", "outputs/batch_results")
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{batch_output_folder}_{timestamp}"
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "log.txt"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.getLogger().level)
            formatter = logging.Formatter(config_manager.get_logging_config().get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Attached batch log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to attach batch file logger: {e}")
        
        # Log comprehensive batch metadata
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING METADATA")
        logger.info("=" * 80)
        logger.info(f"Client Type: {client_type.upper()}")
        logger.info(f"Parent Folder: {parent_folder}")
        logger.info(f"Max Work Orders: {max_work_orders}")
        logger.info(f"Batch Size: {batch_size}")
        
        # Log prompt configuration details
        prompt_subdir = sor_config.get("sor_analysis", {}).get("prompt_subdir", "prompts")
        prompt_version = sor_config.get("sor_analysis", {}).get("prompt_version", None)
        prompt_path = f"{prompt_subdir}/{prompt_version}" if prompt_version else prompt_subdir
        logger.info(f"Prompt Configuration: {prompt_path}")
        
        # Log model configuration details
        if client_type == "azure":
            azure_config = config_manager.get_azure_openai_config()
            deployment_name = azure_config.get("deployment_name", "gpt-4o")
            logger.info(f"Azure Model: {deployment_name}")
            logger.info(f"Azure Endpoint: {azure_config.get('endpoint', 'Not specified')}")
        else:
            aws_config = config_manager.get_aws_config()
            model_id = aws_config.get("bedrock", {}).get("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            region = aws_config.get("aws", {}).get("region", "Not specified")
            logger.info(f"AWS Bedrock Model: {model_id}")
            logger.info(f"AWS Region: {region}")
        
        # Log enabled SOR types
        enabled_sors = processor.get_enabled_sor_types()
        logger.info(f"Enabled SOR Types: {', '.join(enabled_sors)}")
        logger.info("=" * 80)
        
        # Process work orders
        results = await processor.process_work_orders_batch(
            parent_folder=parent_folder,
            max_work_orders=max_work_orders,
            batch_size=batch_size
        )
        
        # Save results (CSV + summary.json) into the batch folder
        saved_files = processor.save_results(results, output_path)
        
        logger.info(f"Processing completed successfully!")
        for ft, fp in saved_files.items():
            logger.info(f"Saved {ft}: {fp}")
        # Handle both Azure (batch_summary) and Bedrock (summary) return structures
        summary_key = 'batch_summary' if 'batch_summary' in results else 'summary'
        logger.info(f"Total work orders processed: {results[summary_key]['total_work_orders']}")
        logger.info(f"Total cost: ${results[summary_key]['total_cost']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to process SOR analysis: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
