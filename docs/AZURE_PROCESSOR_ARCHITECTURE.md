# Azure Unified SOR Processor - Architecture Breakdown

This document provides a comprehensive architectural overview of `scripts/azure_unified_sor_processor.py`, including its design patterns, component interactions, and exact dependencies on the `src/` toolkit.

---

## Overview

The Azure Unified SOR Processor is a sophisticated batch processing system that analyzes electrical work order images using Azure OpenAI's GPT-4o models. It processes multiple work orders concurrently, each containing multiple SOR (Statement of Requirements) types, with intelligent image targeting and comprehensive result aggregation.

---

## Architecture Pattern

### **Async Batch Processing with Concurrency Control**
- **Batch Processing**: Work orders processed in configurable memory-efficient batches
- **Nested Parallelism**: Concurrent work order processing + concurrent SOR analysis within each work order
- **Semaphore-Based Rate Limiting**: Global semaphore controls total concurrent SOR analyses across all work orders
- **Exponential Backoff**: Automatic retry logic for API rate limiting and transient failures

### **Configuration-Driven Design**
- **No Hard-Coded Behavior**: All processing rules defined in YAML configurations
- **Pluggable Prompts**: SOR types added/modified via prompt files, no code changes
- **Fallback Resolution**: Sophisticated prompt loading with version/directory fallbacks
- **Targeted Image Processing**: SOR-specific image selection based on folder mappings

---

## Core Components & Workflow

### **1. Initialization (`__init__`)**
```python
# Key responsibilities:
- Load SOR analysis configuration
- Create shared Azure OpenAI client (efficiency)
- Pre-load ALL prompt configurations (avoids concurrent loading)
- Configure targeted image processing rules
- Set up batch processing parameters
```

**src/ Dependencies Used:**
- `src.clients.azure_openai_client.AzureOpenAIClient` (dynamic import)
- `src.tools.results_table_generator.ResultsTableGenerator`
- `src.config.ConfigManager`

### **2. Work Order Discovery (`discover_work_orders`)**
```python
# Key responsibilities:
- Scan parent directory for work order folders
- Apply exclusion patterns (configurable)
- Count images per work order (recursive search)
- Filter by minimum image requirements
- Sort and limit results
```

**No src/ Dependencies** - Uses standard library Path operations

### **3. Batch Processing (`process_work_orders_batch`)**
```python
# Key responsibilities:
- Memory-efficient batching (configurable batch size)
- Parallel work order processing within each batch
- Global semaphore coordination across all SOR analyses
- Garbage collection between batches
- Result aggregation and summary generation
```

**src/ Dependencies Used:**
- Calls `_process_work_orders_parallel` which uses image loading tools

### **4. Work Order Parallel Processing (`_process_work_orders_parallel`)**
```python
# Key responsibilities:
- Creates global semaphore for SOR-level concurrency control
- Spawns concurrent tasks for all work orders in batch
- Image preparation in thread pools (true parallelism)
- Error isolation per work order
```

**src/ Dependencies Used:**
- `src.tools.image_loader.ImageLoader` (dynamic import in thread pool)

### **5. Image Preparation (`prepare_images_sync`)**
```python
# Key responsibilities:
- Targeted image loading per SOR type
- Folder mapping resolution (SOR-specific folders)
- Fallback to default folder if needed
- Image caching within work order scope
- Format optimization (JPEG for photos)
- Size limiting (20MB for Azure)
```

**src/ Dependencies Used:**
- `src.tools.image_loader.ImageLoader` (full functionality)

### **6. SOR Analysis (`_analyze_sor_type`)**
```python
# Key responsibilities:
- Core analysis function and RAG integration point
- Prompt configuration loading
- Multi-image vs per-image processing modes
- Azure OpenAI API invocation
- Response parsing with fallback handling
- Token/cost tracking
```

**src/ Dependencies Used:**
- Uses pre-loaded prompt configs (via ConfigManager)
- Calls Azure client methods
- Uses client's JSON parsing and fallback parser creation

### **7. Result Generation and Saving**
```python
# Key responsibilities:
- CSV table generation from batch results
- JSON summary creation with comprehensive metadata
- File output management
```

**src/ Dependencies Used:**
- `src.tools.results_table_generator.ResultsTableGenerator` (full table generation)

---

## Detailed src/ Tool Usage

### **ConfigManager (`src.config.ConfigManager`)**
**Usage Pattern**: Constructor injection, used throughout
```python
# Configurations accessed:
- sor_analysis_config.yaml (main configuration)
- Azure OpenAI configuration and pricing
- Model parameters for deployment
- Prompt resolution with fallback logic
```

### **AzureOpenAIClient (`src.clients.azure_openai_client.AzureOpenAIClient`)**
**Usage Pattern**: Single shared instance, dynamic import
```python
# Methods used:
- invoke_model_multi_image() - speed-optimized processing
- invoke_model_per_image() - accuracy-optimized processing  
- repair_json_response() - handles truncated JSON
- parse_json_response() - with fallback parser support
- create_fallback_parser() - SOR-specific parsing
```

### **ImageLoader (`src.tools.image_loader.ImageLoader`)**
**Usage Pattern**: Per-work-order instances, thread pool execution
```python
# Methods used:
- load_images_to_memory() - with timestamp extraction
- encode_images() - with Azure 20MB limit
- Automatic JPEG optimization for photo content
- Format injection (media_type for Azure)
```

### **ResultsTableGenerator (`src.tools.results_table_generator.ResultsTableGenerator`)**
**Usage Pattern**: Single instance, used for final output
```python
# Methods used:
- generate_table_from_batch_results() - converts JSON to DataFrame
- save_table() - CSV output generation
```

### **CLI Utilities (`src.utils`)**
**Usage Pattern**: Module-level functions
```python
# Functions used:
- setup_logging() - configures logging from config
- CLIConfig class - argument parsing and config resolution
```

---

## Concurrency & Performance Design

### **Three-Level Concurrency Model**
1. **Batch Level**: Sequential processing of work order batches
2. **Work Order Level**: Unlimited concurrent work orders within batch
3. **SOR Level**: Global semaphore limits total concurrent SOR analyses

### **Semaphore Calculation**
```python
max_concurrent_sors = max_concurrent_work_orders × max_concurrent_sor_types
# Example: 8 work orders × 8 SOR types = 64 concurrent API calls
```

### **Rate Limiting & Retry Logic**
- **Progressive delays**: API rate limit compliance
- **Exponential backoff**: 1s → 2s → 4s → 8s → 16s → 32s → 60s (max)
- **Error classification**: Retryable vs non-retryable errors
- **Maximum attempts**: 10 retries with detailed logging

### **Memory Management**
- **Batch-based processing**: Prevents memory buildup
- **Explicit garbage collection**: After each batch
- **Image caching**: Per-work-order scope only
- **Result streaming**: No accumulation of large datasets

---

## Configuration Integration Points

### **SOR Analysis Config (`configs/sor_analysis_config.yaml`)**
```yaml
# Controls processor behavior:
- client_type: azure_openai
- prompt_subdir/version: targeted prompt selection
- batch_processing: concurrency and memory settings
- targeted_images: folder mapping and fallback rules
```

### **Azure Config (`configs/azure_config.yaml + models/azure_openai_config.yaml`)**
```yaml
# Controls Azure client:
- endpoint, deployment_name, api_version
- processing.per_image_mode: accuracy vs speed
- pricing configuration for cost calculation
```

### **Prompt Configs (`configs/prompts/Targeted-Prompts/`)**
```yaml
# SOR-specific analysis instructions:
- system_prompt: analysis context
- main_prompt: specific analysis requirements
- Loaded once at initialization, cached throughout
```

---

## Error Handling & Resilience

### **Graceful Degradation**
- **Missing images**: Returns deterministic FAIL result
- **API failures**: Retries with exponential backoff
- **Parse failures**: Uses fallback parsers
- **Work order errors**: Isolated, don't affect other work orders

### **Comprehensive Logging**
- **Analysis context**: SOR type, work order, image details
- **API debugging**: Request/response metrics
- **Performance tracking**: Token usage, costs, timing
- **Error categorization**: Retryable vs fatal errors

### **Result Consistency**
- **Structured output**: Standardized JSON format
- **Token/cost tracking**: Per-SOR and aggregated
- **Metadata preservation**: Timestamps, work order info
- **Status indicators**: Success/failure/error states

---

## Integration Points for Extensions

### **RAG Integration** (Primary Extension Point)
**Location**: `_analyze_sor_type()` method, before Azure API call
**Context Available**: 
- `sor_type`, `evidence_images`, `system_prompt`, `main_prompt`
- Work order information and configuration

### **Custom SOR Types**
**Requirements**: 
- Add prompt YAML file to `configs/prompts/Targeted-Prompts/`
- Configure folder mapping in `sor_analysis_config.yaml`
- No code changes required

### **Alternative Clients**
**Pattern**: Replace `AzureOpenAIClient` import in `__init__`
**Requirements**: Implement same interface as Azure client

### **Output Formats**
**Extension Point**: `save_results()` method
**Current**: CSV + JSON summary
**Extensible**: Additional formats via ResultsTableGenerator

---

## Performance Characteristics

### **Throughput Optimization**
- **Shared client instance**: Reduces connection overhead
- **Parallel processing**: Maximizes API utilization
- **Targeted images**: Reduces irrelevant processing
- **Prompt pre-loading**: Eliminates I/O during processing

### **Memory Efficiency**
- **Batch processing**: Configurable memory usage
- **Image streaming**: No persistent storage of encoded data
- **Result aggregation**: Minimal memory footprint
- **Garbage collection**: Explicit cleanup between batches

### **Cost Optimization**
- **Comprehensive tracking**: Token and cost monitoring
- **Format optimization**: JPEG for photos (smaller payloads)
- **Size limiting**: Automatic image compression
- **Processing modes**: Speed vs accuracy trade-offs

---

This architecture enables robust, scalable SOR analysis with clear extension points for RAG integration and other enhancements while maintaining high performance and reliability.
