# RISC-V Parameter Extraction Pipeline

Automated extraction of architectural parameters from RISC-V specification text using LLMs with schema-guided validation.

## Overview

This pipeline supports two extraction modes:

1. **One-Shot Extraction** - Direct extraction using proprietary models (Claude, GPT-4, etc.)
2. **RAG-based Extraction** - Retrieval-Augmented Generation using open-source models (Qwen, Llama, DeepSeek)

Both modes extract structured parameters from RISC-V specification snippets and output YAML files conforming to predefined JSON schemas.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [One-Shot Extraction (Proprietary Models)](#one-shot-extraction-proprietary-models)
  - [RAG-based Extraction (Open-Source Models)](#rag-based-extraction-open-source-models)
- [Output Format](#output-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Quick Start

### For Proprietary Models (Claude, GPT-4/5, Gemini, Grok)

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY='your-key-here'
export OPENAI_API_KEY='your-key-here'  # optional

# Create prompts
mkdir -p prompts
echo "You are an expert in RISC-V parameter extraction..." > prompts/system_prompt_1.txt
echo "Extract parameters from: [your snippet]" > prompts/user_prompt_2.txt

# Run extraction (tests multiple models)
python -m src.one_shot.run_one_shot
```

### For Open-Source Models (Qwen, Llama, DeepSeek)

```bash
# Install dependencies
pip install -r requirements.txt

# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b

# Run extraction
python -m src.rag.main \
  -snippets snippets/snippet_1.txt \
  -schemas schemas \
  -dir outputs
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- (For RAG mode) Ollama runtime

### Step 1: Clone and Install Dependencies

```bash
git clone <repository-url>
cd risc-v-parameter-extraction
pip install -r requirements.txt

# For one-shot mode, also install:
pip install litellm python-dotenv
```

### Step 2: Choose Your Mode

#### For One-Shot Mode (Proprietary Models)

Set up API credentials:

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY='sk-ant-...'

# For GPT-4 (OpenAI)
export OPENAI_API_KEY='sk-...'
```

#### For RAG Mode (Open-Source Models)

Install Ollama and pull models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service (run in separate terminal or background)
ollama serve

# Pull models (choose one or more)
ollama pull qwen2.5:7b      # Recommended: Fast and accurate
ollama pull llama3.1:8b     # Alternative: Good reasoning
ollama pull deepseek-r1:8b  # Alternative: Strong analytical capabilities
```

---

## Directory Structure

```
risc-v-parameter-extraction/
├── src/
│   ├── one_shot/              # One-shot extraction for proprietary models
│   │   └── run_one_shot.py
│   ├── rag/                   # RAG-based extraction for open-source models
│   │   ├── main.py            # Pipeline entry point
│   │   ├── config.py          # Configuration settings
│   │   ├── rag_engine.py      # RAG orchestration
│   │   ├── vector_store.py    # Schema vectorization
│   │   ├── llm_processor.py   # LLM interface
│   │   ├── schema_loader.py   # Schema parsing
│   │   ├── yaml_generator.py  # Output generation
│   │   ├── evaluator.py       # Metrics computation
│   │   └── test_pipeline.py   # Installation verification
│   └── utils/                 # Utility scripts
│       ├── analyse_yaml_outputs.py
│       └── extract_schema_keys.py
├── schemas/                   # JSON schema definitions
│   ├── inst_schema.json
│   ├── csr_schema.json
│   └── [other schemas]
├── snippets/                  # Input specification text
│   ├── snippet_1.txt
│   └── [other snippets]
├── outputs/                   # Generated YAML files
└── requirements.txt
```

---

## Usage

## One-Shot Extraction (Proprietary Models)

One-shot extraction uses proprietary models (Claude, GPT-4/5, Gemini, Grok, DeepSeek) via the LiteLLM framework for direct parameter extraction without retrieval.

### Setup

#### Required Environment Variables

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY='sk-ant-api03-...'

# For GPT-4/5 (OpenAI)
export OPENAI_API_KEY='sk-...'

# For Gemini (Google)
export GEMINI_API_KEY='...'

# For Grok (xAI)
export XAI_API_KEY='...'

# Load from .env file (optional)
# Create a .env file with your keys:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

#### Install LiteLLM

```bash
pip install litellm python-dotenv
```

### Basic Usage

The one-shot extraction reads prompts from files and runs extraction across multiple models:

```bash
# Create prompt directories
mkdir -p prompts outputs

# Create system prompt (prompt engineering instructions)
cat > prompts/system_prompt_1.txt << 'EOF'
You are an expert in RISC-V architecture parameter extraction.
Extract structured parameters from the provided specification text.
Output valid YAML conforming to the provided JSON schemas.
EOF

# Create user prompt (snippet + schemas)
cat > prompts/user_prompt_2.txt << 'EOF'
Extract parameters from this RISC-V specification:

[Your snippet text here]

Use these schemas as reference:
[Schema content or references]
EOF

# Run extraction
python -m src.one_shot.run_one_shot
```

### Configuration

Edit `src/one_shot/run_one_shot.py` to configure models to test:

```python
models_to_test = [
    "claude-4.5-sonnet",      # Claude Sonnet 4.5
    "gemini-3-flash",         # Gemini 3 Flash
    "gemini-3-pro",           # Gemini 3 Pro
    "gpt-5",                  # GPT-5
    "grok-4.1"                # Grok 4.1
]
```

### Supported Models

Via LiteLLM, you can use:

- **Claude**: `claude-4.5-sonnet`, `claude-opus-4.5`, `claude-sonnet-3-5`
- **GPT**: `gpt-5`, `gpt-4-turbo`, `gpt-4`
- **Gemini**: `gemini-3-pro`, `gemini-3-flash`, `gemini-2.0-flash`
- **Grok**: `grok-4.1`, `grok-3`
- **DeepSeek**: `deepseek-chat`, `deepseek-coder`

Full list: https://docs.litellm.ai/docs/providers

### Prompt Files

The script expects two prompt files:

1. **System Prompt** (`prompts/system_prompt_1.txt`)
   - Role definition
   - Task instructions
   - Output format requirements
   
2. **User Prompt** (`prompts/user_prompt_2.txt`)
   - Specification snippet
   - Schema references
   - Specific extraction requirements

### Output Structure

Each model generates a timestamped YAML file:

```
outputs/
├── claude-4.5-sonnet_test_results_20240209_143052.yaml
├── gemini-3-flash_test_results_20240209_143105.yaml
├── gpt-5_test_results_20240209_143118.yaml
├── grok-4.1_test_results_20240209_143131.yaml
└── all_results_20240209_143145.yaml
```

### Example Output

```
Testing Agentic Framework Across Models
==================================================

Testing: claude-4.5-sonnet
Response:
- name: csr_encoding_space_size
  description: The bit width of the CSR address encoding space
  type: Integer
  value: 12
...
Tokens Used: 1247
Saved parsed YAML to: outputs/claude-4.5-sonnet_test_results_20240209_143052.yaml

Testing: gpt-5
Response:
- name: csr_address_width
  description: Width of CSR address field
...
Tokens Used: 1134
Saved parsed YAML to: outputs/gpt-5_test_results_20240209_143118.yaml

Consolidated results saved to: outputs/all_results_20240209_143145.yaml
```

### Consolidated Results

The `all_results_*.yaml` file contains:

```yaml
test_metadata:
  timestamp: "2024-02-09T14:31:45"
  total_models_tested: 5
  models:
    - claude-4.5-sonnet
    - gemini-3-flash
    - gpt-5
    
results:
  - content: "[YAML output]"
    model: claude-4.5-sonnet
    usage:
      prompt_tokens: 823
      completion_tokens: 424
      total_tokens: 1247
    tested_at: "2024-02-09T14:30:52"
```

### Advanced Configuration

#### Custom Temperature and Tokens

Edit in `run_one_shot.py`:

```python
result = test_agentic_framework(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    model=model,
    temperature=0.3,      # Lower for more deterministic
    max_tokens=2000       # Increase for longer outputs
)
```

#### Add Custom Models

```python
models_to_test = [
    "claude-4.5-sonnet",
    "anthropic/claude-opus-4.5",  # Explicit provider
    "openai/gpt-5",
    "your-custom-model"
]
```

---

## RAG-based Extraction (Open-Source Models)

### Basic Usage

```bash
python -m src.rag.main \
  -snippets snippets/snippet_1.txt \
  -schemas schemas \
  -dir outputs
```

### Options

```bash
python -m src.rag.main \
  -snippets <path>        # Snippet file or directory
  -schemas <path>         # Schema directory
  -dir <path>            # Output directory
  -model <name>          # Model: qwen, llama, deepseek (default: qwen)
  -k <number>            # Top-K schemas to retrieve (default: 5)
```

### Model Selection

```bash
# Use Qwen (fastest, recommended)
python -m src.rag.main -snippets snippets -schemas schemas -dir outputs -model qwen

# Use Llama (good reasoning)
python -m src.rag.main -snippets snippets -schemas schemas -dir outputs -model llama

# Use DeepSeek (strong analytical)
python -m src.rag.main -snippets snippets -schemas schemas -dir outputs -model deepseek
```

### Adjusting Schema Retrieval

```bash
# Retrieve more schemas for complex snippets
python -m src.rag.main \
  -snippets snippets/complex_snippet.txt \
  -schemas schemas \
  -dir outputs \
  -k 10

# Fewer schemas for focused extraction
python -m src.rag.main \
  -snippets snippets/simple_snippet.txt \
  -schemas schemas \
  -dir outputs \
  -k 3
```

### Batch Processing

```bash
# Process all snippets with all models
for model in qwen llama deepseek; do
    python -m src.rag.main \
      -snippets snippets \
      -schemas schemas \
      -dir outputs \
      -model $model
done
```

### Example Output

```
============================================================
RISC-V Parameter Extraction Pipeline
============================================================
Model: qwen2.5:7b
Top-K Schemas: 5
Output Directory: outputs
============================================================

Loading schemas...
Loaded 22 schemas
Building vector store...
Vector store ready
Initializing LLM (qwen2.5:7b)...
LLM ready

Processing 1 snippet(s)...

Processing: snippet_1.txt
  Primary Category: csr_schema
  Retrieved Schemas: csr_schema, inst_schema, config_schema, param_schema, prm_schema
  Saved: qwen_snippet_1_v1.yaml

============================================================
Model: qwen2.5:7b | Snippet: snippet_1.txt
============================================================
├── Total Parameters: 6
├── Unique Parameters: 6
├── Novel Parameters: 4
└── Duplicates: 0

Extracted Parameters:
  1. csr_encoding_space_size
  2. csr_address_width
  3. csr_read_write_accessibility_encoding
  4. csr_privilege_level_encoding
  5. csr_total_addressable_count
  6. csr_address_mapping_convention
============================================================
```

### Testing Installation

```bash
# Verify RAG pipeline setup
python -m src.rag.test_pipeline
```

---

## Output Format

### YAML Structure

Both modes generate YAML files with the following parameter structure:

```yaml
- name: csr_encoding_space_size
  description: The bit width of the CSR address encoding space
  type: Integer
  unit: bits
  value: 12
  source_quote: "The standard RISC-V ISA sets aside a 12-bit encoding space"
  rationale: Defines the fundamental addressing capacity for CSRs
  scope: ISA-wide specification
  constraints:
    - Fixed at 12 bits in standard RISC-V ISA

- name: csr_read_write_accessibility_encoding
  description: Encoding scheme for CSR read/write permissions
  type: Enum
  possible_values:
    - "00: read/write"
    - "01: read/write"
    - "10: read/write"
    - "11: read-only"
  source_quote: "The top two bits (csr[11:10]) indicate whether the register is read/write"
  rationale: Enables hardware to enforce access permissions based on address
  scope: Per CSR instance
```

### Required Fields

- `name`: Unique parameter identifier
- `description`: What the parameter represents
- `type`: Data type (Integer, String, Enum, Bitfield, etc.)
- `source_quote`: Direct quote from specification
- `rationale`: Architectural significance
- `scope`: Application scope (ISA-wide, per-instance, etc.)

### Optional Fields

- `unit`: Measurement unit (bits, bytes, cycles, etc.)
- `value`: Specific value if mentioned
- `constraints`: List of constraints
- `possible_values`: Valid values for enums
- `properties`: Sub-properties for composite types
- `additional_requirements`: Extra notes

### File Naming

**One-Shot Mode:**
```
outputs/
├── claude_snippet_1.yaml
├── claude_snippet_2.yaml
└── gpt4_snippet_1.yaml
```

**RAG Mode:**
```
outputs/
├── qwen_snippet_1_v1.yaml
├── qwen_snippet_1_v2.yaml    # Auto-versioning
├── llama_snippet_1_v1.yaml
└── deepseek_snippet_1_v1.yaml
```

---

## Evaluation Metrics

Both modes compute the following metrics:

### Metrics Explained

- **Total Parameters**: All extracted parameters
- **Unique Parameters**: Non-duplicate parameters
- **Novel Parameters**: Parameters NOT found in any schema (new discoveries)
- **Duplicates**: Repeated parameter extractions

### Novel Parameter Criteria

A parameter is considered "novel" if:
- NOT explicitly defined in any loaded schema
- Architecturally significant (not trivial)
- Derived from specification text
- Non-repetitive

### Example Novel Parameters

**Good Novel Parameters:**
- `csr_address_mapping_convention` - Design philosophy
- `csr_total_addressable_count` - Derived from encoding space
- `privilege_level_access_matrix` - Composite relationship

**Poor Novel Parameters:**
- `csr_address` - Too vague, already in schemas
- `register_type` - Generic, not specific
- `bit_value` - Duplicates existing parameters

### Viewing Metrics

**One-Shot Mode:**
```bash
# Metrics printed to console after extraction
# Use analysis utilities:
python -m src.utils.analyse_yaml_outputs outputs/
```

**RAG Mode:**
```bash
# Metrics printed automatically during extraction
# Detailed report:
============================================================
Model: qwen2.5:7b | Snippet: snippet_1.txt
============================================================
├── Total Parameters: 6
├── Unique Parameters: 6
├── Novel Parameters: 4
└── Duplicates: 0
============================================================
```

---

## Visualization and Analysis

The pipeline includes powerful visualization tools to compare extraction results across models and snippets.

### Basic Analysis

```bash
# Analyze all YAML files in outputs directory
python -m src.utils.analyse_yaml_outputs outputs/

# With novelty scoring (requires schema directory)
python -m src.utils.analyse_yaml_outputs outputs/ -s schemas

# Specify custom output directory for visualizations
python -m src.utils.analyse_yaml_outputs outputs/ -s schemas -o visualizations/
```

### Command Options

```bash
python -m src.utils.analyse_yaml_outputs <directory> [options]

Arguments:
  directory              Directory containing YAML output files

Options:
  -s, --schema <path>   Schema directory for novelty scoring
  -o, --output <path>   Output directory for visualizations (default: /home/claude)
  -q, --quiet           Suppress verbose output
```

### Generated Visualizations

The analysis tool generates comprehensive visualizations:

#### 1. Comparison Plots (`comparison_plots.png`)

A multi-panel figure showing:
- **Quantity Comparison**: Total items extracted by each model
- **Average Quality**: Average fields per item
- **Unique Keys**: Total distinct field types used
- **Max Quality**: Maximum fields in any single item
- **Novelty Score**: Novel parameters discovered (if schemas provided)
- **Quality vs Quantity Scatter**: Trade-off visualization
- **Novelty Ratio**: Proportion of novel parameters
- **By Snippet Breakdown**: Performance per snippet

#### 2. Summary Table (`summary_table.png`)

Professional table with:
- Model names
- Quantity (total items)
- Average quality (fields per item)
- Max quality
- Unique field types
- Novelty score and ratio (if available)

#### 3. CSV Export (`analysis_results.csv`)

Structured data for further analysis:
```csv
Model,Quantity,Avg_Quality,Max_Quality,Unique_Keys,Novelty_Score,Novelty_Ratio
claude_S1-4.5-snippet_1,6,8.67,12,15,4,0.267
qwen_snippet_1_v1,6,7.83,11,12,3,0.250
```

#### 4. Detailed Report (`detailed_report.txt`)

Text report with:
- Full metrics for each model
- Complete list of all field types
- Novel parameters identified
- Standard schema matches

### Example Analysis Session

```bash
# Complete analysis workflow
$ python -m src.utils.analyse_yaml_outputs outputs/ -s schemas -o analysis_output/

Found 4 YAML files to analyze:

  - claude_S1-4.5-snippet_1
  - qwen_snippet_1_v1
  - llama_snippet_1_v1
  - deepseek_snippet_1_v1

Extracting standard parameters from 22 schema files:
  - csr_schema.json
    Found 47 unique keys
  - inst_schema.json
    Found 38 unique keys
  ...

Total unique standard parameters: 156

Analyzing: claude_S1-4.5-snippet_1...
  Novelty: 4 novel params (26.7%)
Analyzing: qwen_snippet_1_v1...
  Novelty: 3 novel params (25.0%)
...

================================================================================
ANALYSIS RESULTS
================================================================================

Definitions:
  Quantity:        Total number of items/parameters extracted
  Avg Quality:     Average number of unique fields per item
  Max Quality:     Maximum unique fields in any single item
  Unique Keys:     Total number of distinct field types used
  Novelty Score:   Number of parameters NOT in standard schema
  Novelty Ratio:   Proportion of novel to total parameters

================================================================================

claude_S1-4.5-snippet_1:
  Quantity:        6
  Avg Quality:     8.67
  Max Quality:     12
  Unique Keys:     15
  Novelty Score:   4
  Novelty Ratio:   26.67%
  Standard Match:  11
  Field types:     name, description, type, value, unit, source_quote, rationale, scope
                   ... and 7 more
...

Generating visualizations...
Saved to: analysis_output/comparison_plots.png
Summary table saved to: analysis_output/summary_table.png
Detailed results saved to: analysis_output/analysis_results.csv
Detailed report saved to: analysis_output/detailed_report.txt
Schema keys reference saved to: analysis_output/standard_schema_keys.txt

================================================================================
ANALYSIS COMPLETE!
================================================================================

All outputs saved to: analysis_output/
```

### Understanding the Metrics

**Quantity Metrics:**
- **Total Items**: More is generally better, but quality matters
- Shows thoroughness of extraction

**Quality Metrics:**
- **Avg Quality**: Higher means more detailed parameters
- **Max Quality**: Shows best-case extraction capability
- **Unique Keys**: Diversity of field types used

**Novelty Metrics:**
- **Novelty Score**: New parameters discovered (not in schemas)
- **Novelty Ratio**: Percentage of novel vs standard parameters
- Higher novelty = more discovery potential

### Interpreting Visualizations

#### Quality vs Quantity Scatter Plot

- **Top-right quadrant**: High quantity + high quality (ideal)
- **Top-left**: High quality but low quantity (selective)
- **Bottom-right**: High quantity but low quality (broad but shallow)
- **Bottom-left**: Low on both metrics

#### Novelty Analysis

- **High novelty ratio (>30%)**: Model discovering many new patterns
- **Medium novelty (15-30%)**: Balanced extraction
- **Low novelty (<15%)**: Primarily extracting standard parameters

### Comparing Models

```bash
# Generate comparison after running multiple models
python -m src.rag.main -snippets snippets/snippet_1.txt -schemas schemas -dir outputs -model qwen
python -m src.rag.main -snippets snippets/snippet_1.txt -schemas schemas -dir outputs -model llama
python -m src.rag.main -snippets snippets/snippet_1.txt -schemas schemas -dir outputs -model deepseek
python -m src.one_shot.run_one_shot  # Runs Claude, GPT, Gemini, etc.

# Analyze all results together
python -m src.utils.analyse_yaml_outputs outputs/ -s schemas -o model_comparison/
```

### Advanced Analysis

#### Extract Schema Keys Only

```bash
# View all parameters defined in schemas
python -m src.utils.extract_schema_keys schemas/

# Save to file
python -m src.utils.extract_schema_keys schemas/ -o schema_parameters.txt
```

#### Novelty Deep Dive

```bash
# Run analysis with schema comparison
python -m src.utils.analyse_yaml_outputs outputs/ -s schemas -o analysis/

# Check detailed report for novel parameters
cat analysis/detailed_report.txt | grep -A 20 "Novel Parameters"
```

#### Custom Grouping

The analysis automatically groups results by:
- **Model family**: All Claude outputs, all Qwen outputs, etc.
- **Snippet**: All extractions from snippet_1, snippet_2, etc.

This allows you to see:
- How different models handle the same snippet
- How the same model handles different snippets

### Visualization Tips

1. **Large datasets**: If analyzing many files (>10), increase figure size in code
2. **Color coding**: Each model family has a consistent color across plots
3. **Export formats**: Modify code to save as PDF for publications (`plt.savefig(..., format='pdf')`)
4. **Custom plots**: Use `analysis_results.csv` with pandas/matplotlib for custom analysis

### Batch Analysis Workflow

```bash
#!/bin/bash
# Complete extraction and analysis pipeline

# 1. Extract with all models
for model in qwen llama deepseek; do
    python -m src.rag.main -snippets snippets/ -schemas schemas -dir outputs -model $model
done

python -m src.one_shot.run_one_shot

# 2. Analyze and visualize
python -m src.utils.analyse_yaml_outputs outputs/ -s schemas -o final_analysis/

# 3. Review outputs
open final_analysis/comparison_plots.png
open final_analysis/summary_table.png
cat final_analysis/detailed_report.txt
```

---

## Troubleshooting

### One-Shot Mode Issues

#### API Key Not Set

```bash
# Error: "API key not found"
export ANTHROPIC_API_KEY='your-key-here'

# Verify
echo $ANTHROPIC_API_KEY
```

#### Rate Limiting

```bash
# Error: "Rate limit exceeded"
# Wait and retry, or reduce batch size
python -m src.one_shot.run_one_shot \
  --snippets snippets/snippet_1.txt \
  --schemas schemas \
  --output outputs
```

### RAG Mode Issues

#### Ollama Not Running

```bash
# Error: "Ollama not available"
# Check status
ollama list

# Start service
ollama serve

# In new terminal, verify
ollama list
```

#### Model Not Found

```bash
# Error: "Model 'qwen2.5:7b' not found"
ollama pull qwen2.5:7b

# List available models
ollama list
```

#### Schema Directory Not Found

```bash
# Error: "No schemas found in: schemas"
# Check directory
ls schemas/*.json | head -5

# Use absolute path
python -m src.rag.main \
  -snippets snippets/snippet_1.txt \
  -schemas /absolute/path/to/schemas \
  -dir outputs
```

#### YAML Parsing Errors

```bash
# Error: "Failed to parse YAML"
# LLM output may be malformed - try again (non-deterministic)
python -m src.rag.main -snippets snippets/snippet_1.txt -schemas schemas -dir outputs

# Try different model
python -m src.rag.main -snippets snippets/snippet_1.txt -schemas schemas -dir outputs -model llama
```

#### Low Novel Parameter Count

```bash
# Increase schema retrieval
python -m src.rag.main \
  -snippets snippets/snippet_1.txt \
  -schemas schemas \
  -dir outputs \
  -k 10

# Try model with better reasoning
python -m src.rag.main \
  -snippets snippets/snippet_1.txt \
  -schemas schemas \
  -dir outputs \
  -model deepseek
```

### General Issues

#### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

#### Permission Errors

```bash
# Create output directory
mkdir -p outputs
chmod 755 outputs
```

---

## Advanced Configuration

### Customizing Schema Retrieval (RAG Mode)

Edit `src/rag/config.py`:

```python
class Config:
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Default top-K retrieval
    DEFAULT_TOP_K = 5
    
    # LLM models
    MODEL_CONFIGS = {
        'qwen': 'qwen2.5:7b',
        'llama': 'llama3.1:8b',
        'deepseek': 'deepseek-r1:8b'
    }
```

### Customizing Prompts

**One-Shot Mode:**
Edit prompt in `src/one_shot/run_one_shot.py`

**RAG Mode:**
Edit system prompt in `src/rag/rag_engine.py`

### Adding New Schemas

```bash
# Add JSON schema to schemas/
cp new_schema.json schemas/

# Pipeline will automatically load it
python -m src.rag.main -snippets snippets -schemas schemas -dir outputs
```

### Batch Processing Scripts

```bash
# Process all snippets with all models (RAG)
#!/bin/bash
for snippet in snippets/*.txt; do
    for model in qwen llama deepseek; do
        python -m src.rag.main \
          -snippets "$snippet" \
          -schemas schemas \
          -dir outputs \
          -model $model
    done
done
```

```bash
# Compare one-shot vs RAG
#!/bin/bash
snippet="snippets/snippet_1.txt"

# One-shot
python -m src.one_shot.run_one_shot \
  --snippets "$snippet" \
  --schemas schemas \
  --output outputs

# RAG with multiple models
for model in qwen llama deepseek; do
    python -m src.rag.main \
      -snippets "$snippet" \
      -schemas schemas \
      -dir outputs \
      -model $model
done
```

---


## Tips and Best Practices

### Model Selection

**Use One-Shot (Claude/GPT-4) when:**
- Highest accuracy is required
- Processing critical specifications
- Budget allows API costs
- Complex reasoning needed

**Use RAG (Qwen/Llama/DeepSeek) when:**
- Processing large volumes of text
- Cost-effectiveness is priority
- Local/offline processing needed
- Iterative experimentation required

### Optimization Tips

1. **First run downloads embedding model** (~80MB) - subsequent runs are faster
2. **Use qwen for speed** - fastest of the three open-source models
3. **Increase top-k for complex snippets** - use `-k 7` or `-k 10`
4. **Version auto-increments** - safe to run multiple times
5. **Novel params indicate quality** - check metrics for discovery rate
6. **Batch similar snippets** - process CSR snippets together, instruction snippets together

## Contributing

### Adding New Features

1. For one-shot mode: Edit files in `src/one_shot/`
2. For RAG mode: Edit files in `src/rag/`
3. Test changes with `python -m src.rag.test_pipeline`

### Schema Development

Add new schemas to `schemas/` directory following JSON Schema format.


## Acknowledgments

Built for automated extraction of architectural parameters from RISC-V specifications using:
- OpenAI embeddings (sentence-transformers)
- Ollama for local LLM inference
- Anthropic Claude API for proprietary extraction
- LangChain for RAG orchestration