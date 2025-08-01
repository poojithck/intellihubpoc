# IntelliHub SOR POC

**Configurable, async image analysis with AWS Bedrock (Claude 3.5 Sonnet) and Pillow.**  
Analyze fuse cartridges, cracks, meter readings, or any scenario—just add a YAML prompt, no code changes required.

---

## 🚀 Quickstart

**Requirements:**  
- Python **3.11+** (managed by uv, see below)
- [uv](https://github.com/astral-sh/uv) (install: `pip install uv` or `brew install uv`)

**Setup:**

```bash
# 1. Clone the repository
git clone https://github.com/your-org/intellihubSORpoc.git
cd intellihubSORpoc

# 2. Install Python (if needed) and create a virtual environment
uv python install 3.11
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies (from pyproject.toml/uv.lock)
uv pip install -r pyproject.toml
```

**To update dependencies:**
```bash
uv pip install --upgrade -r pyproject.toml
uv pip freeze > uv.lock
```

---

## 🏃 Usage

**Unified SOR Processing (Recommended):**

Process all work orders with all SOR types:
```bash
uv run scripts/unified_sor_processor.py --parent-folder path/to/work_orders
```

Process specific SOR types only:
```bash
uv run scripts/unified_sor_processor.py --parent-folder path/to/work_orders --sor-list AsbestosBagAndBoard,FuseReplacement
```

List available SOR types:
```bash
uv run scripts/unified_sor_processor.py --list-sors
```

**Grid-based (multi-image) analysis example:**

```python
from src.tools import ImageAnalyzer, ImageGridder
from src.config import ConfigManager

config_manager = ConfigManager()
analyzer = ImageAnalyzer(config_manager, "AsbestosBagAndBoard")
gridder = ImageGridder(config_manager)

# Analyze all images in a folder as grids (multi-image LLM analysis)
result = analyzer.analyze_image_grids("path/to/images", gridder)
print(result["parsed_response"])
```

**Key Features:**
- **Efficient Processing**: Single grid generation per work order, shared across all SOR types
- **Configuration-Driven**: All SOR types defined in `configs/sor_analysis_config.yaml`
- **Comprehensive Output**: JSON, CSV, Excel tables with summary statistics
- **Parallel Processing**: Concurrent work order processing for speed

Add your own analysis type in **3 steps**—see [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).

---

## 🗂️ Project Structure

```
configs/         # All YAML config (prompts, models, app, AWS, SOR analysis)
docs/            # Guides and developer docs
scripts/         # Unified SOR processor and CLI tools
src/
  clients/       # BedrockClient (AWS API)
  config/        # ConfigManager (YAML loader)
  tools/         # ImageAnalyzer, WorkOrderProcessor, ImageGridder, ResultsTableGenerator
  utils/         # Logging, CLI helpers
tests/           # Simple test/demo scripts
pyproject.toml   # Project metadata and dependencies
uv.lock          # uv lockfile for reproducible installs
```

---

## 🧑‍💻 Development

```bash
# Install dev tools (one-time)
uv pip install ruff mypy pytest

# Lint
ruff check src tests scripts

# Type-check
mypy src

# Run tests (requires AWS creds or mocks)
pytest
```

---

## 📝 Configuration

- **Prompts & analysis logic:** `configs/prompts/*.yaml`
- **Model params:** `configs/models/claude_config.yaml`
- **Image processing:** `configs/app_config.yaml`
- **AWS/Bedrock:** `configs/aws_config.yaml`

See [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md) for how to add new analysis types—no code required.

---

## 🧑‍💻 Best Practices

- **Use the Unified SOR Processor** for production batch processing of work orders
- **Configuration-driven approach**: Define new SOR types in `configs/sor_analysis_config.yaml`
- **Efficient processing**: The system automatically optimizes grid generation and reuse
- **Comprehensive output**: Get JSON, CSV, Excel, and summary reports in one run
- Keep all analysis logic in YAML prompts for easy maintenance and updates

---

## ⚡ About uv

- **uv** is a modern, fast Python package manager and environment tool.
- It manages Python versions, virtual environments, dependencies, and lockfiles.
- See [uv documentation](https://github.com/astral-sh/uv) for advanced usage.
