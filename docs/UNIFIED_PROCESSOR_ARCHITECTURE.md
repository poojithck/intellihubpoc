# Unified SOR Processor - Architecture Breakdown

This document provides a comprehensive architectural overview of `scripts/unified_sor_processor.py`, including its design patterns, component interactions, and exact dependencies on the `src/` toolkit. This processor represents the **AWS Bedrock implementation** with both targeted image processing and legacy grid-based analysis modes.

---

## Overview

The Unified SOR Processor is a dual-mode batch processing system that analyzes electrical work order images using AWS Bedrock's Claude 3.5 Sonnet models. It supports both **targeted image processing** (SOR-specific folder mapping) and **legacy grid-based analysis** (single grid per work order), with intelligent mode switching and comprehensive result aggregation.

---

## Architecture Pattern

### **Hybrid Processing Architecture**
- **Dual Mode Support**: Targeted images OR grid-based processing (configurable)
- **Async Batch Processing**: Memory-efficient batching with concurrency control
- **Nested Parallelism**: Concurrent work order processing + concurrent SOR analysis within each work order
- **Shared Resource Optimization**: Single grid generation per work order (legacy mode)
- **Global Semaphore Control**: Rate limiting across all concurrent SOR analyses

### **Configuration-Driven Design**
- **Mode Toggle**: `targeted_mode` flag switches between processing approaches
- **No Hard-Coded Behavior**: All processing rules defined in YAML configurations
- **Pluggable Prompts**: SOR types added/modified via prompt files, no code changes
- **Fallback Resolution**: Sophisticated prompt loading with version/directory fallbacks

---

## Core Components & Workflow

### **1. Initialization (`__init__`)**
```python
# Key responsibilities:
- Load SOR analysis configuration
- Create shared AWS Bedrock client (efficiency)
- Initialize ImageGridder for legacy mode
- Pre-load ALL prompt configurations (avoids concurrent loading)
- Configure dual-mode processing rules (targeted vs grid)
- Set up batch processing parameters
```

**src/ Dependencies Used:**
- `src.clients.bedrock_client.BedrockClient` (dynamic import)
- `src.tools.results_table_generator.ResultsTableGenerator`
- `src.tools.image_gridder.ImageGridder`
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
- Calls `_process_work_orders_parallel` which orchestrates image preparation

### **4. Work Order Parallel Processing (`_process_work_orders_parallel`)**
```python
# Key responsibilities:
- Creates global semaphore for SOR-level concurrency control
- Spawns concurrent tasks for all work orders in batch
- Dual-mode image preparation (targeted OR grid) in thread pools
- Error isolation per work order
```

**src/ Dependencies Used:**
- `src.tools.image_loader.ImageLoader` (dynamic import for targeted mode)
- `src.tools.image_gridder.ImageGridder` (for legacy grid mode)

### **5. Image Preparation (`prepare_images_sync`)**
```python
# Key responsibilities:
- DUAL MODE OPERATION:
  * Targeted Mode: SOR-specific folder mapping with fallbacks
  * Grid Mode: Single grid generation shared across all SORs
- Image caching within work order scope
- Format optimization (JPEG for targeted, PNG for grids)
- Size limiting (5MB for AWS Bedrock)
```

**src/ Dependencies Used:**
- **Targeted Mode**: `src.tools.image_loader.ImageLoader` (full functionality)
- **Grid Mode**: `src.tools.image_gridder.ImageGridder` (grid creation and encoding)

### **6. SOR Analysis (`_analyze_sor_type`)**
```python
# Key responsibilities:
- Core analysis function and RAG integration point
- Prompt configuration loading with system + main prompt composition
- AWS Bedrock API invocation (multi-image mode)
- Response parsing with fallback handling
- Token/cost tracking
```

**src/ Dependencies Used:**
- Uses pre-loaded prompt configs (via ConfigManager)
- Calls Bedrock client methods
- Uses BedrockClient's static JSON parsing and fallback parser creation

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
- AWS/Bedrock configuration and pricing
- Model parameters for Claude models
- Prompt resolution with fallback logic
```

### **BedrockClient (`src.clients.bedrock_client.BedrockClient`)**
**Usage Pattern**: Single shared instance, dynamic import
```python
# Methods used:
- invoke_model_multi_image() - all SOR processing
- Static methods:
  * repair_json_response() - handles truncated JSON
  * create_fallback_parser() - SOR-specific parsing
- parse_json_response() - with fallback parser support
```

### **ImageLoader (`src.tools.image_loader.ImageLoader`)**
**Usage Pattern**: Per-work-order instances (TARGETED MODE ONLY)
```python
# Methods used:
- load_images_to_memory() - with timestamp extraction
- encode_images() - with AWS 5MB limit
- JPEG optimization for photo content
- Format injection (media_type for Bedrock)
```

### **ImageGridder (`src.tools.image_gridder.ImageGridder`)**
**Usage Pattern**: Single instance (GRID MODE ONLY)
```python
# Methods used:
- create_grids() - generates single grid per work order
- encode_grids() - with AWS 5MB limit
- Memory management (grid cleanup after encoding)
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

## Dual-Mode Processing Architecture

### **Targeted Mode (`targeted_mode: true`)**
```python
# Characteristics:
- SOR-specific folder mapping
- Multiple images per SOR type
- Fallback to default folder
- Per-SOR image limits
- Higher accuracy (context-specific images)
```

**Image Flow:**
1. Map SOR type to specific folders via `folder_mapping`
2. Load images from mapped folders
3. Fallback to `default_folder_name` if no specific images
4. Apply `max_images_per_sor` limit
5. JPEG encoding for photo optimization

### **Grid Mode (`targeted_mode: false`)**
```python
# Characteristics:
- Single grid generation per work order
- Shared grid across ALL SOR types
- Memory efficient (one grid, multiple analyses)
- Grid-based visual context
- Legacy compatibility
```

**Image Flow:**
1. Create single grid from all work order images
2. PNG encoding for grid preservation
3. Shared grid used for all SOR type analyses
4. Memory cleanup after encoding

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
- **AWS-Specific Error Handling**: ThrottlingException, TooManyRequestsException
- **Exponential backoff**: 1s → 2s → 4s → 8s → 16s → 32s → 60s (max)
- **Maximum attempts**: 10 retries with detailed logging
- **Service-specific detection**: ServiceUnavailableException, InternalServerError

### **Memory Management**
- **Mode-Specific Optimization**:
  - **Targeted**: Per-SOR image loading, folder caching
  - **Grid**: Single grid generation, immediate cleanup
- **Batch-based processing**: Prevents memory buildup
- **Explicit garbage collection**: After each batch

---

## Configuration Integration Points

### **SOR Analysis Config (`configs/sor_analysis_config.yaml`)**
```yaml
# Controls processor behavior:
- client_type: aws_bedrock
- targeted_images.enabled: true/false (mode selection)
- folder_mapping: SOR-to-folder mappings (targeted mode)
- batch_processing: concurrency and memory settings
```

### **AWS Config (`configs/aws_config.yaml + models/claude_config.yaml`)**
```yaml
# Controls Bedrock client:
- bedrock.default_model: Claude model selection
- aws.region: AWS region configuration
- pricing: cost calculation for Claude models
```

### **Prompt Configs (`configs/prompts/Targeted-Prompts/`)**
```yaml
# SOR-specific analysis instructions:
- system_prompt: analysis context and strict definitions
- main_prompt: specific analysis requirements
- Composed as: system + "---" + main (if system exists)
```

---

## Key Differences from Azure Processor

### **Processing Modes**
| Feature | Unified (AWS) | Azure |
|---------|---------------|-------|
| **Modes** | Targeted OR Grid | Targeted Only |
| **Image Selection** | SOR-specific OR shared grid | SOR-specific only |
| **Fallback** | Default folder OR grid generation | Default folder only |
| **Size Limit** | 5MB (AWS Bedrock) | 20MB (Azure OpenAI) |
| **Format** | JPEG/PNG based on mode | JPEG optimized |

### **Client Architecture**
| Feature | Unified (AWS) | Azure |
|---------|---------------|-------|
| **API Client** | BedrockClient | AzureOpenAIClient |
| **Processing Modes** | Multi-image only | Multi-image OR per-image |
| **Authentication** | AWS credentials | API key |
| **Rate Limiting** | AWS-specific errors | HTTP-specific errors |

### **Grid Processing**
| Feature | Unified (AWS) | Azure |
|---------|---------------|-------|
| **Grid Support** | Yes (legacy mode) | No |
| **Grid Usage** | Shared across all SORs | N/A |
| **Grid Tool** | ImageGridder integration | Not used |

---

## Error Handling & Resilience

### **Graceful Degradation**
- **Missing images**: Returns deterministic FAIL result
- **Grid creation failure**: Work order-level error isolation
- **API failures**: AWS-specific retry logic with exponential backoff
- **Parse failures**: Uses fallback parsers
- **Mode fallback**: Grid mode if targeted folder mapping fails

### **AWS-Specific Error Handling**
- **ThrottlingException**: Automatic retry with backoff
- **ServiceUnavailableException**: Retry logic
- **Timeout errors**: Connection and read timeout handling
- **Region availability**: AWS region-specific error handling

### **Dual-Mode Error Isolation**
- **Targeted mode failures**: Falls back to empty image list (FAIL result)
- **Grid mode failures**: Work order marked as error
- **Mixed mode**: Per-SOR error isolation within work order

---

## Integration Points for Extensions

### **RAG Integration** (Primary Extension Point)
**Location**: `_analyze_sor_type()` method, before Bedrock API call
**Context Available**: 
- `sor_type`, `images` (targeted OR grid), `prompt_text` (composed)
- Work order information and configuration
- Mode information (targeted vs grid)

### **Custom Processing Modes**
**Extension Point**: `prepare_images_sync()` method
**Pattern**: Add new mode alongside targeted/grid modes
**Requirements**: Update mode detection and image preparation logic

### **Alternative Grid Strategies**
**Extension Point**: Grid mode section in `prepare_images_sync()`
**Current**: Single grid per work order
**Extensible**: Multiple grids, SOR-specific grids, dynamic grid sizing

### **Hybrid Mode Implementation**
**Potential**: Combine targeted + grid processing
**Location**: Mode selection logic in `prepare_images_sync()`
**Benefits**: Targeted accuracy + grid context

---

## Performance Characteristics

### **Mode-Specific Performance**
| Metric | Targeted Mode | Grid Mode |
|--------|---------------|-----------|
| **Memory Usage** | Higher (multiple images per SOR) | Lower (single grid shared) |
| **Processing Speed** | Slower (more API calls) | Faster (fewer unique images) |
| **Accuracy** | Higher (context-specific) | Variable (depends on grid quality) |
| **Cost** | Higher (more tokens) | Lower (fewer unique images) |

### **Throughput Optimization**
- **Shared client instance**: Reduces connection overhead
- **Mode-appropriate optimization**: JPEG for photos, PNG for grids
- **Parallel processing**: Maximizes API utilization
- **Prompt pre-loading**: Eliminates I/O during processing

### **Cost Optimization**
- **Mode selection**: Grid mode for cost-sensitive scenarios
- **Comprehensive tracking**: Token and cost monitoring
- **Size limiting**: Automatic image compression (5MB AWS limit)
- **Format optimization**: Mode-appropriate image formats

---

## Legacy Compatibility & Migration Path

### **Grid Mode as Migration Bridge**
- **Purpose**: Maintains compatibility with existing grid-based workflows
- **Use Case**: Organizations with established grid analysis processes
- **Benefits**: Minimal workflow disruption during transition

### **Gradual Migration Strategy**
1. **Start with Grid Mode**: Existing workflows unchanged
2. **Test Targeted Mode**: Compare accuracy on subset of SOR types
3. **Selective Migration**: Move high-value SOR types to targeted mode
4. **Full Targeted**: Complete migration when validated

### **Configuration-Driven Transition**
```yaml
# Easy mode switching via configuration
targeted_images:
  enabled: false  # Start with grid mode
  # Later: enabled: true for targeted mode
```

---

This architecture enables flexible, scalable SOR analysis with clear mode selection and extension points for RAG integration, while maintaining backward compatibility with grid-based processing workflows.
