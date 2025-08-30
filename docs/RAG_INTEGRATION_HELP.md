## RAG Integration Help (Azure Path) — Architecture and Integration Points

This document explains the key components involved in the Azure SOR pipeline and where a Retrieval Augmented Generation (RAG) step can be inserted. It is intended to help a collaborator quickly understand the code paths and branch off to add a RAG module. It does not prescribe how to build the RAG itself.

---

## 1) Core Components (Azure)

### `scripts/azure_unified_sor_processor.py`
- Orchestrates batch processing of work orders.
- Selects targeted images per SOR type (config-driven) and preloads prompts (versioned).
- Performs per-SOR analysis in `_analyze_sor_type(...)`.
- Aggregates results, tracks token usage/cost, writes CSV and JSON summaries.

### `src/clients/azure_openai_client.py`
- Wraps Azure OpenAI Chat Completions API.
- Two invocation modes:
  - `invoke_model_multi_image(...)`: multi-image in one call (speed)
  - `invoke_model_per_image(...)`: one image per call (accuracy-first)
- Adds text metadata per image (filename, timestamps) to improve grounding.

### `src/config/config_manager.py`
- Loads prompts with fallback rules and versioning (`get_prompt_config`).
- Provides Azure config and model param accessors.

### `configs/sor_analysis_config.yaml`
- Toggles client type and SOR enablement.
- Sets prompt directory/version and batch settings.

---

## 2) How the Azure Flow Works (High-Level)

1. Discover work orders (folders) and identify images relevant to each enabled SOR.
2. Load SOR prompts (system + main) from the configured versioned directory.
3. Call Azure OpenAI with the images and prompts:
   - Either multi-image in one request or per-image aggregation.
4. Parse model output into structured JSON; compute per-work-order and per-batch summaries (including tokens/costs).

This flow is entirely configuration-driven; adding/modifying SOR behavior is done by editing YAML prompts and settings.

---

## 3) RAG Integration Point

Insert RAG immediately before the model invocation in:

`scripts/azure_unified_sor_processor.py` → `_analyze_sor_type(...)`

At this moment the processor has:
- `sor_type` (e.g., `FuseReplacement`),
- evidence images (already selected/encoded),
- `system_prompt` and `main_prompt` strings.

The RAG step would retrieve small, relevant text from local sources and attach it to the user prompt (e.g., as a brief “Relevant Context” section). The Azure client remains unchanged; downstream parsing, metrics, and output formats are unaffected.

Conceptual flow:
```
Targeted Images → (Optional) RAG Context → Prompt = System + Main (+ Context) → Azure Model → Parse → Aggregate
```

---

## 4) Configuration Touchpoints

If you want the RAG step to be optional and configurable, use `configs/sor_analysis_config.yaml` to:
- Add a `rag_settings` block (e.g., enabled flag, path to local documents, per-SOR toggles).
- Keep the rest of the pipeline unchanged by gating the context injection with this flag.

No other config files need to move; prompt versioning and model parameter presets remain as-is.

---

## 5) Key Constraints and Cues

- Keep appended context short to avoid displacing core instructions.
- Maintain the existing separation of `system_prompt` and user `main_prompt`.
- Preserve image metadata lines; they help tie references back to filenames/timestamps.
- Do not alter the Azure client interface; inject context only in the prompt composition step.

---

## 6) Where to Look in the Codebase

- Prompt loading and resolution order: `src/config/config_manager.py` (`get_prompt_config`)
- Azure request packaging: `src/clients/azure_openai_client.py` (`invoke_model_multi_image`, `invoke_model_per_image`)
- SOR analysis entry point: `scripts/azure_unified_sor_processor.py` (`_analyze_sor_type`)
- Batch outputs and summaries: `scripts/azure_unified_sor_processor.py` (save and summary helpers)

These locations provide the minimal hooks needed to add a RAG retrieval step without broader refactors.

