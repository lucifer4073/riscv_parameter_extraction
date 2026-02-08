# RISC-V Parameter Extraction RAG Pipeline

A RAG-based system for extracting architectural parameters from RISC-V specification snippets using Ollama models.

## Features

- Multi-model support: qwen2.5:7b, llama3.1:8b, deepseek-r1-distill-llama-8b
- FAISS vector store for schema retrieval
- Automatic novel parameter discovery
- YAML output with auto-versioning
- Evaluation metrics (total, unique, novel parameters)

## Installation

```bash
pip install -r requirements.txt
```

### Prerequisites

1. **Ollama** must be installed and running:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull qwen2.5:7b
   ollama pull llama3.1:8b
   ollama pull deepseek-r1-distill-llama-8b
   ```

2. **Schema files**: Ensure you have the schema directory with JSON files

## Usage

### Basic Usage (Single Snippet)

```bash
python main.py \
  -snippets snippet_1.txt \
  -dir ./outputs \
  -schemas ./schemas
```

### Specify Model

```bash
python main.py \
  -snippets snippet_1.txt \
  -dir ./outputs \
  -schemas ./schemas \
  -model llama
```

Available models: `qwen` (default), `llama`, `deepseek`

### Process Multiple Snippets

```bash
python main.py \
  -snippets ./snippets_folder \
  -dir ./outputs \
  -schemas ./schemas \
  -model deepseek
```

### Custom Top-K Retrieval

```bash
python main.py \
  -snippets snippet_1.txt \
  -dir ./outputs \
  -schemas ./schemas \
  -k 3
```

## Arguments

- `-snippets` (required): Path to .txt snippet file or directory
- `-dir` (required): Output directory for YAML files
- `-schemas` (required): Directory containing JSON schema files
- `-model` (optional): Model to use (qwen/llama/deepseek, default: qwen)
- `-k` (optional): Number of schemas to retrieve (default: 5)

## Output

### File Naming

Output files follow the format: `{model}_{snippet_name}_v{version}.yaml`

Examples:
- `qwen_snippet_1_v1.yaml`
- `llama_snippet_1_v1.yaml`
- `deepseek_snippet_1_v2.yaml` (auto-versioned)

### YAML Structure

Each extracted parameter includes:

**Required fields:**
- `name`: Parameter name
- `description`: Parameter description
- `type`: Data type
- `source_quote`: Quote from snippet
- `rationale`: Architectural significance
- `scope`: Application scope

**Optional fields:**
- `unit`: Measurement unit
- `constraints`: Parameter constraints
- `possible_values`: Valid values (enums)
- `value`: Specific value
- `properties`: Sub-properties
- `interface_type`: Interface type
- `additional_requirements`: Extra notes

## Evaluation Metrics

The pipeline outputs:
- **Total Parameters**: Count of all extracted parameters
- **Unique Parameters**: Count of distinct parameters
- **Novel Parameters**: Parameters not in schemas
- **Duplicates**: Repeated parameters

## Project Structure

```
rag_pipeline/
├── main.py              # CLI entry point
├── config.py            # Configuration constants
├── schema_loader.py     # JSON schema parser
├── vector_store.py      # FAISS vector store
├── llm_processor.py     # Ollama interface
├── rag_engine.py        # RAG logic & prompts
├── yaml_generator.py    # YAML output handler
├── evaluator.py         # Metrics computation
└── requirements.txt     # Dependencies
```

## Example

```bash
# Process snippet with qwen model
python main.py \
  -snippets examples/snippet_1.txt \
  -dir outputs \
  -schemas schemas/

# Output:
# ============================================================
# Model: qwen2.5:7b | Snippet: snippet_1.txt
# ============================================================
# ├── Total Parameters: 8
# ├── Unique Parameters: 7
# ├── Novel Parameters: 4
# └── Duplicates: 1
```

## Troubleshooting

**Ollama not available:**
```bash
# Start Ollama service
ollama serve
```

**Model not found:**
```bash
# Pull the required model
ollama pull qwen2.5:7b
```

**YAML parsing errors:**
- Check LLM response format
- Try different temperature settings in `llm_processor.py`

## Notes

- First run downloads sentence-transformers model (~80MB)
- Vector store is rebuilt each run (can be cached for production)
- Novel parameter detection uses fuzzy matching against schema definitions