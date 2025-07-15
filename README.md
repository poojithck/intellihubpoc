# IntelliHub SOR Proof-of-Concept

Config-driven, async image-analysis toolkit powered by **AWS Bedrock** multimodal models (Claude 3.5 Sonnet) and Pillow.  
Analyse fuse cartridges, cracks, meter readings (and any future scenario) with **zero code changes** – just add a YAML prompt.

---

## ✨ Key Features

* **Config-first design** – prompts, model parameters and pricing live in `configs/`
* **Single generic analyzer** – `ImageAnalyzer` handles every use-case via configuration
* **Async & concurrency-limited** – processes many images without overloading Bedrock
* **Automatic cost tracking** – see token counts & USD cost per analysis
* **Pluggable fallback parsing** – robustly extracts answers even from non-JSON output
* **High-quality image preprocessing** – smart down-scaling, format conversion & base64 encoding

---

## 📦 Installation

```bash
# 1. Clone repository
$ git clone https://github.com/your-org/intellihubSORpoc.git
$ cd intellihubSORpoc

# 2. (Optional) create virtual env using uv (recommended)
$ uv venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install project and dependencies with uv
$ uv pip install -e .
```

> Python 3.11+ is recommended.

---

## 🚀 Quick Start

Analyse fuse cartridges in the default `artefacts/` folder:

```bash
python scripts/fuse_analysis.py
```

Analyse meter-reading images in a custom folder:

```bash
python scripts/meter_reading.py --path /path/to/my/images  # (coming soon via CLI helper)
```

Add your own analysis type in **3 steps** – see [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md).

---

## 🗂️ Repository Layout

```
configs/                # YAML config (prompts, models, app, AWS)
docs/                   # Developer & user documentation
scripts/                # CLI entry-points for each analysis type
src/
 ├── clients/           # BedrockClient – model invocation & cost calc
 ├── config/            # ConfigManager – loads & caches YAML config
 ├── tools/             # ImageAnalyzer & ImageLoader
 └── utils/             # Logging helpers, future CLI utilities
tests/                  # Simple demo / smoke tests
```

---

## 🛠️ Development

```bash
# Lint
uv pip install ruff mypy pytest  # first-time setup of dev tools
ruff check src tests scripts

# Type-check
mypy src

# Test (needs mocked AWS creds)
pytest -q
```

### Git Hooks
Add pre-commit hooks for `ruff`, `mypy`, `pytest` to catch issues early.
