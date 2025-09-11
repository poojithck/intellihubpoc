#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.rag.pipeline import RAGPipeline
from src.tools.image_loader import ImageLoader

async def test_rag():
    # Initialize
    config_manager = ConfigManager()
    rag_pipeline = RAGPipeline(config_manager)
    
    # Check repository statistics
    stats = rag_pipeline.get_statistics()
    print(f"RAG Repository Stats: {stats}")
    
    # Load a test work order image
    loader = ImageLoader("path/to/test/work_order/folder")
    images = loader.load_images_to_memory(single=False)
    encoded_images = loader.encode_images(images)
    
    # Get prompt config for MeterConsolidationE4
    prompt_config = config_manager.get_prompt_config("MeterConsolidationE4")
    
    # Enhance with RAG
    enhanced_prompt_config, combined_images = rag_pipeline.enhance_sor_analysis(
        sor_type="MeterConsolidationE4",
        prompt_config=prompt_config,
        work_order_images=encoded_images
    )
    
    print(f"Original images: {len(encoded_images)}")
    print(f"Combined images (with references): {len(combined_images)}")
    print("RAG enhancement successful!")
    
    # Check if prompts were augmented
    if enhanced_prompt_config.get('rag_enhanced'):
        print("✓ Prompts were successfully augmented with reference examples")

if __name__ == "__main__":
    asyncio.run(test_rag())