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

Analyze fuse cartridges (default images in `artefacts/`):

```bash
uv run scripts/fuse_analysis.py
```

Analyze meter readings (custom folder):

```bash
uv run scripts/meter_reading.py  # edit script to set your folder, or add CLI arg
```

Add your own analysis type in **3 steps**—see [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).

---

## 🗂️ Project Structure

```
configs/         # All YAML config (prompts, models, app, AWS)
docs/            # Guides and developer docs
scripts/         # CLI entry-points for each analysis type
src/
  clients/       # BedrockClient (AWS API)
  config/        # ConfigManager (YAML loader)
  tools/         # ImageAnalyzer, ImageLoader
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

## ⚡ About uv

- **uv** is a modern, fast Python package manager and environment tool.
- It manages Python versions, virtual environments, dependencies, and lockfiles.
- See [uv documentation](https://github.com/astral-sh/uv) for advanced usage.
