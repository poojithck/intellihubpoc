# IntelliHub SOR POC

Configurable, asynchronous image analysis of electrical work orders using Large Language Models (LLMs) with support for both AWS Bedrock and Azure OpenAI. The system is prompt- and configuration-driven, enabling new SOR (Statement of Requirements) analyses to be added without changing code.

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

## 🏃 Usage (Quick)

Unified SOR processing (AWS Bedrock path):
```bash
uv run scripts/unified_sor_processor.py
```

Unified SOR processing (Azure OpenAI path):
```bash
uv run scripts/azure_unified_sor_processor.py
```

Useful options (where supported):
```bash
--test-mode                      # small, fast run
--max-work-orders 10             # limit number processed
--parent-folder path/to/WOs      # override source folder
--sor-list A,B,C                 # process specific SORs only
--list-sors                      # list enabled SOR types
```

Add a new analysis type via YAML prompts—see [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).

---

## 🗂️ Project Structure

```
configs/         # All YAML config (prompts, models, app, AWS, SOR analysis)
docs/            # Guides and developer docs
scripts/         # Unified SOR processor and CLI tools
src/
  clients/       # BedrockClient (AWS), AzureOpenAIClient (Azure), KeyVault
  config/        # ConfigManager (YAML loader)
  tools/         # ImageAnalyzer, WorkOrderProcessor, ImageGridder, ResultsTableGenerator
  utils/         # Logging, CLI helpers
tests/           # Simple test/demo scripts
pyproject.toml   # Project metadata and dependencies
uv.lock          # uv lockfile for reproducible installs
```

---

## 🧭 High-Level Architecture

The system operates as a batch processor for “work orders”, each containing folders of images. For each SOR type (e.g., `FuseReplacement`, `MeterConsolidationE4`), the pipeline:

1. Discovers work orders and gathers targeted evidence images per SOR (config-driven).
2. Loads the SOR’s prompts (system and main) from versioned YAML.
3. Invokes the selected LLM client (AWS Bedrock or Azure OpenAI) with single or multiple images.
4. Parses the model response into a structured JSON object (with robust fallbacks), then aggregates results.
5. Writes outputs to CSV and JSON summaries, with token and cost metrics.

Key design points:
- Prompt/versioning system allows non-code changes to analysis logic.
- Concurrency: parallel work order and per-SOR processing.
- Image handling: targeted selection or multi-image inputs; size limits enforced.
- Metrics: token counts and cost estimates collected per request and per batch.

For an overview of adding RAG context on the Azure path, see [docs/RAG_INTEGRATION_GUIDE.md](docs/RAG_INTEGRATION_GUIDE.md).

## 🔌 Major Components

- `scripts/unified_sor_processor.py` (AWS Bedrock)
  - Orchestrates batch runs with the Bedrock client.
  - Targeted image preparation and multi-image request packaging.

- `scripts/azure_unified_sor_processor.py` (Azure OpenAI)
  - Mirrors the unified flow for Azure; supports per-image or multi-image modes.
  - Clean injection point for optional RAG prompt augmentation.

- `src/clients/bedrock_client.py` and `src/clients/azure_openai_client.py`
  - API wrappers for model invocation and response parsing.
  - Insert image metadata (filenames, timestamps) and handle token/cost reporting.

- `src/config/config_manager.py`
  - Loads YAML configuration (prompts, models, app, SOR analysis) with fallback rules.

- `src/tools/`
  - `ImageAnalyzer`, `ImageGridder`, `ResultsTableGenerator`, `work_order_processor` helpers.

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

- Prompts & analysis logic: `configs/prompts/**` (with versioned subfolders)
- Model params (AWS): `configs/models/claude_config.yaml`
- Model params (Azure): `configs/models/azure_openai_config.yaml`
- Image processing: `configs/app_config.yaml`
- AWS/Bedrock: `configs/aws_config.yaml`
- Azure/OpenAI: `configs/azure_config.yaml`

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
