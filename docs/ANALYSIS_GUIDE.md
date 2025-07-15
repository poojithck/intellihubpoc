# Image Analysis Guide

This guide explains how to create new image analysis types using the tools and configuration system.

## Quick Start: Adding a New Analysis Type

### 1. Create a Prompt Configuration

Create a new YAML file in `configs/prompts/` with your analysis type name:

```yaml
# configs/prompts/my_analysis.yaml
my_analysis:
  system_prompt: |
    You are an expert in analyzing [your domain] images.
    
  main_prompt: |
    [Your specific question about the image]

    Please provide your response in the following JSON format:
    {
        "answer": "Yes" or "No",
        "note": "One sentence explanation"
    }

    [Your specific criteria for Yes/No answers]
  
  fallback_keywords:
    positive: ["yes", "keyword1", "keyword2"]
    negative: ["no", "keyword3", "keyword4"]
  
  response_format:
    required_fields: ["answer", "note"]
    answer_values: ["Yes", "No"]
    
  model_config: "analysis"  # Use precise analysis settings
```

### 2. Create a Script

Create a new script in `scripts/`:

```python
#!/usr/bin/env python3
"""
My Analysis Script

Description of what this script analyzes.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ConfigurableAnalyzer
from src.config import ConfigManager

def setup_logging(config_manager: ConfigManager) -> None:
    """Setup logging based on configuration."""
    logging_config = config_manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

async def main():
    """Main function to run the analysis."""
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Setup logging
        setup_logging(config_manager)
        
        # Initialize analyzer with your analysis configuration
        analyzer = ConfigurableAnalyzer(config_manager, "my_analysis")
        
        # Analyze images from a specific folder or use default
        results = await analyzer.analyze_images("path/to/your/images")
        
        # Display results
        analyzer.display_results(results, "My Analysis")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Run Your Analysis

```bash
python -m scripts.my_analysis
```

That's it! No other code changes required.

## Available Tools

### ConfigurableAnalyzer
- **Purpose**: Generic analyzer that works with any prompt configuration
- **Usage**: `ConfigurableAnalyzer(config_manager, "analysis_type")`
- **Methods**:
  - `analyze_images(folder_path)` - Analyze images from a folder
  - `display_results(results, title)` - Display formatted results

### FuseAnalyzer
- **Purpose**: Specialized analyzer for fuse cartridge analysis
- **Usage**: `FuseAnalyzer(config_manager)`
- **Methods**:
  - `analyze_fuse_images(folder_path)` - Analyze fuse images

### ImageAnalyzer
- **Purpose**: Base class for all analyzers
- **Usage**: Usually inherited by specialized analyzers
- **Methods**:
  - `load_and_process_images(folder)` - Load, resize, and encode images
  - `analyze_images_batch(images, prompt, params)` - Batch analyze images

## Configuration Options

### Model Settings
Modify `configs/models/claude_config.yaml` to add new model parameter sets:

```yaml
specialized_configs:
  my_precise_analysis:
    max_tokens: 150
    temperature: 0.0  # Very deterministic
  my_creative_analysis:
    max_tokens: 500
    temperature: 0.8  # More creative
```

Then reference it in your prompt config:
```yaml
model_config: "my_precise_analysis"
```

### Image Processing
Modify `configs/app_config.yaml` to change default image processing:

```yaml
image_processing:
  default_resize:
    width: 1200
    height: 1200
  default_format: "JPEG"
```

### Paths
Set default image folders in `configs/app_config.yaml`:

```yaml
paths:
  default_image_folder: "artefacts/MyImages"
```

## Examples

See the existing examples:
- `scripts/fuse_analysis.py` - Fuse cartridge analysis
- `scripts/crack_analysis.py` - Crack detection analysis
- `configs/prompts/fuse_analysis.yaml` - Fuse analysis configuration
- `configs/prompts/crack_analysis.yaml` - Crack analysis configuration

## Architecture

```
configs/
├── app_config.yaml          # App settings, image processing
├── aws_config.yaml          # AWS/Bedrock configuration
├── models/
│   └── claude_config.yaml   # Model parameters
└── prompts/
    ├── fuse_analysis.yaml   # Fuse analysis prompts
    └── crack_analysis.yaml  # Crack analysis prompts

src/
├── tools/
│   ├── image_analyzer.py    # Analysis tools
│   └── image_loader.py      # Image processing
├── clients/
│   └── bedrock_client.py    # AI client
└── config/
    └── config_manager.py    # Configuration management

scripts/
├── fuse_analysis.py         # Fuse analysis script
└── crack_analysis.py        # Crack analysis script
```

The system is designed to be:
- **Config-driven**: No code changes needed for new analysis types
- **Reusable**: Tools can be used across different analysis types
- **Extensible**: Easy to add new models, prompts, or analysis types
- **Maintainable**: Clean separation of concerns 