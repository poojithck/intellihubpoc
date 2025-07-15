# Image Analysis Guide

This guide explains how to create new image analysis types using the tools and configuration system.

## Quick Start: Adding a New Analysis Type

> The codebase now uses **one generic class** – `ImageAnalyzer`.  
> No inheritance or wrapper classes are required.

### 1. Create / Edit Prompt Configuration

Create a YAML file in `configs/prompts/` named after your analysis type (e.g. `my_analysis.yaml`):

```yaml
system_prompt: |
  You are an expert at analysing <your domain> images.

main_prompt: |
  <Your instructions & JSON schema>

fallback_keywords:
  positive: ["yes", "pulled", "present"]
  negative: ["no", "not", "absent"]

response_format:
  required_fields: ["answer", "note"]
  answer_values: ["Yes", "No"]

model_config: "analysis"   # Which parameter set from configs/models
```

### 2. Run via Generic Script Template

```python
#!/usr/bin/env python3
import asyncio, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import ImageAnalyzer
from src.config import ConfigManager
from src.utils import setup_logging

async def main():
    cm = ConfigManager()
    setup_logging(cm)
    analyzer = ImageAnalyzer(cm, "my_analysis")
    results = await analyzer.analyze_images("/path/to/images")
    print(analyzer.format_results(results))

if __name__ == "__main__":
    asyncio.run(main())
```

No new Python classes are required – simply pass the **analysis type** matching your YAML filename.

---

## Creating Model Parameter Presets
(Add in `configs/models/claude_config.yaml`):

```yaml
my_low_cost:
  max_tokens: 100
  temperature: 0.4
```

Reference it in your prompt YAML via `model_config: "my_low_cost"`.

---

## Updating Existing Scripts

Existing scripts `fuse_analysis.py`, `crack_analysis.py`, `meter_reading.py` already follow this pattern. Use them as blueprints for future scripts.

---

## Image Processing Defaults

Defaults live in `configs/app_config.yaml` – update `image_processing` keys to change resize limit or default format.

```yaml
image_processing:
  default_resize:
    width: 1200
    height: 1200
  default_format: "JPEG"
``` 