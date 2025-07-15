#!/usr/bin/env python3
"""
Configuration Test Script

This script tests the configuration system to ensure all configs are loaded correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager

def test_config_system():
    """Test the configuration system."""
    print("Testing Configuration System")
    print("=" * 50)
    
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        
        # Test listing available configs
        print("\n1. Available Configurations:")
        available_configs = config_manager.list_available_configs()
        print(f"   Root configs: {available_configs['root']}")
        print(f"   Subdirectory configs: {available_configs['subdirectories']}")
        
        # Test AWS config
        print("\n2. AWS Configuration:")
        aws_config = config_manager.get_aws_config()
        print(f"   Region: {aws_config['aws']['region']}")
        print(f"   Default Model: {aws_config['bedrock']['default_model']}")
        
        # Test Bedrock client config
        print("\n3. Bedrock Client Configuration:")
        bedrock_config = config_manager.get_bedrock_client_config()
        print(f"   Model ID: {bedrock_config['model_id']}")
        print(f"   Region: {bedrock_config['region_name']}")
        
        # Test model configuration
        print("\n4. Model Configuration:")
        model_config = config_manager.get_model_config()
        print(f"   Model ID: {model_config['model_id']}")
        print(f"   Default params: {model_config['default_params']}")
        
        # Test specialized model params
        print("\n5. Analysis Model Parameters:")
        analysis_params = config_manager.get_model_params(config_type="analysis")
        print(f"   Analysis params: {analysis_params}")
        
        # Test prompt configuration
        print("\n6. Fuse Analysis Prompt Configuration:")
        prompt_config = config_manager.get_prompt_config("fuse_analysis")
        print(f"   Model config type: {prompt_config.get('model_config')}")
        print(f"   Required fields: {prompt_config['response_format']['required_fields']}")
        
        # Test pricing configuration
        print("\n7. Pricing Configuration:")
        pricing_config = config_manager.get_pricing_config()
        print(f"   Input price per 1K: ${pricing_config['input_price_per_1k']}")
        print(f"   Output price per 1K: ${pricing_config['output_price_per_1k']}")
        
        # Test app configuration
        print("\n8. Application Configuration:")
        app_config = config_manager.get_app_config()
        print(f"   App name: {app_config['app']['name']}")
        print(f"   Version: {app_config['app']['version']}")
        
        # Test image processing config
        print("\n9. Image Processing Configuration:")
        img_config = config_manager.get_image_processing_config()
        print(f"   Default resize: {img_config['default_resize']}")
        print(f"   Default format: {img_config['default_format']}")
        print(f"   Supported formats: {len(img_config['supported_formats'])} formats")
        
        # Test logging configuration
        print("\n10. Logging Configuration:")
        logging_config = config_manager.get_logging_config()
        print(f"   Level: {logging_config['level']}")
        print(f"   Format: {logging_config['format'][:50]}...")
        
        print("\n" + "=" * 50)
        print("✅ All configuration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        raise

if __name__ == "__main__":
    test_config_system() 