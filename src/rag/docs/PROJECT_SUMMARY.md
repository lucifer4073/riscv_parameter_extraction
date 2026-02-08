# RISC-V Parameter Extraction RAG Pipeline - Project Summary

## Overview

A production-ready RAG (Retrieval-Augmented Generation) pipeline for extracting architectural parameters from RISC-V specification snippets using Ollama LLMs with FAISS vector retrieval.

**Version:** 1.0  
**Models Supported:** qwen2.5:7b, llama3.1:8b, deepseek-r1-distill-llama-8b  
**Primary Use Case:** Extract both schema-defined and novel architectural parameters from RISC-V docs

## Key Features

✅ **Multi-Model Support** - Compare extraction across 3 different LLMs  
✅ **Intelligent RAG** - FAISS vector store with top-k schema retrieval  
✅ **Novel Parameter Discovery** - Identifies parameters beyond schema definitions  
✅ **Auto-Versioning** - Prevents file overwrites with automatic version increment  
✅ **Comprehensive Metrics** - Track total, unique, and novel parameter counts  
✅ **Batch Processing** - Handle multiple snippets sequentially  
✅ **Clean Codebase** - Simple, well-documented, production-ready code

## Project Structure

```
rag_pipeline/
├── Core Pipeline
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration constants
│   ├── schema_loader.py     # JSON schema parser
│   ├── vector_store.py      # FAISS embedding & retrieval
│   ├── llm_processor.py     # Ollama interface
│   ├── rag_engine.py        # RAG logic & prompting
│   ├── yaml_generator.py    # YAML output handler
│   └── evaluator.py         # Metrics computation
│
├── Documentation
│   ├── README.md            # Main documentation
│   ├── QUICKSTART.md        # 5-minute setup guide
│   ├── ARCHITECTURE.md      # System design details
│   ├── USAGE_EXAMPLE.md     # Real example walkthrough
│   └── PROJECT_SUMMARY.md   # This file
│
├── Utilities
│   ├── test_pipeline.py     # Structure validation tests
│   ├── example.sh           # Usage examples
│   └── requirements.txt     # Python dependencies
│
└── Requirements
    └── schemas/             # User-provided JSON schemas (22 files)
```

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Ollama (qwen/llama/deepseek) | Parameter extraction |
| **Embeddings** | sentence-transformers | Schema vectorization |
| **Vector Store** | FAISS | Similarity search |
| **RAG Framework** | LangChain | Document retrieval |
| **Output Format** | YAML | Structured parameters |
| **Language** | Python 3.10+ | Implementation |

## Installation

### Quick Install
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 3. Pull models
ollama pull qwen2.5:7b
```

### Full Installation (All Models)
```bash
pip install -r requirements.txt
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull deepseek-r1-distill-llama-8b
```

## Usage

### Basic Command
```bash
python main.py -snippets <snippet_path> -dir <output_dir> -schemas <schema_dir>
```

### Common Patterns

**Single snippet, default model:**
```bash
python main.py -snippets snippet_1.txt -dir outputs -schemas schemas
```

**Specific model:**
```bash
python main.py -snippets snippet_1.txt -dir outputs -schemas schemas -model llama
```

**Batch processing:**
```bash
python main.py -snippets snippets_folder -dir outputs -schemas schemas
```

**All models comparison:**
```bash
for model in qwen llama deepseek; do
    python main.py -snippets snippet_1.txt -dir outputs -schemas schemas -model $model
done
```

## Input/Output Specifications

### Input Requirements

**Snippets:**
- Format: `.txt` files
- Content: RISC-V specification text
- Location: Single file or directory

**Schemas:**
- Format: JSON (JSON Schema Draft-07)
- Count: 22 schema files
- Required fields: `properties`, `$defs`, `description`

### Output Format

**Filename Pattern:** `{model}_{snippet_name}_v{version}.yaml`

**YAML Structure:**
```yaml
- name: parameter_name
  description: What this parameter represents
  type: Integer|String|Enum|Boolean|...
  source_quote: "Direct quote from snippet"
  rationale: Why this is architecturally significant
  scope: System-wide|Per-core|Per-cache|...
  # Optional fields
  unit: bytes|cycles|...
  constraints: [list of constraints]
  possible_values: [valid values]
  value: specific_value
  properties: {sub-properties}
```

## Evaluation Metrics

The pipeline computes and displays:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Total Parameters** | All extracted items | `len(parameters)` |
| **Unique Parameters** | Distinct parameter names | `len(set(param.name))` |
| **Novel Parameters** | Not in schemas | Fuzzy match against schemas |
| **Duplicates** | Repeated parameters | `total - unique` |

### Example Output
```
============================================================
Model: qwen2.5:7b | Snippet: snippet_1.txt
============================================================
├── Total Parameters: 8
├── Unique Parameters: 7
├── Novel Parameters: 4
└── Duplicates: 1

Extracted Parameters:
  1. csr_encoding_space_size
  2. csr_address_width
  3. csr_read_write_accessibility
  ...
============================================================
```

## Design Principles

### 1. Simplicity
- Clean, minimal codebase
- Standard Python patterns
- No over-engineering

### 2. Modularity
- Each component has single responsibility
- Easy to extend or replace modules
- Clear interfaces between components

### 3. Robustness
- Graceful error handling
- Informative warnings
- No silent failures

### 4. Configurability
- Easy model switching
- Adjustable retrieval parameters
- Flexible output options

## Key Implementation Details

### RAG Strategy

**Retrieval Process:**
1. Embed all schemas using sentence-transformers
2. Store embeddings in FAISS index
3. For each snippet, retrieve top-k most similar schemas
4. Primary category = highest scoring schema

**Prompt Engineering:**
- System prompt includes retrieved schema definitions
- Emphasizes both schema-defined and novel parameters
- Requires source quotes for all parameters
- Specifies flexible YAML output format

### Novel Parameter Detection

**Algorithm:**
```python
for param in extracted:
    param_normalized = normalize(param.name)
    is_novel = True
    for schema_param in all_schema_params:
        if fuzzy_match(param_normalized, schema_param):
            is_novel = False
            break
    if is_novel:
        novel_count += 1
```

**Normalization:** Lowercase, remove underscores/hyphens  
**Matching:** Exact match or substring containment

### Auto-Versioning

**Logic:**
```python
version = 1
while output_path.exists():
    version += 1
    output_path = f"{model}_{snippet}_v{version}.yaml"
```

**Result:** Never overwrites existing files

## Performance Characteristics

### Timing (Approximate)

| Operation | First Run | Subsequent Runs |
|-----------|-----------|-----------------|
| Embedding model download | 30-60s | 0s (cached) |
| Schema loading | 2-5s | 2-5s |
| Vector store build | 5-10s | 5-10s |
| LLM inference (per snippet) | 10-30s | 10-30s |

### Scalability

- **Schemas:** Linear O(n)
- **Snippets:** Linear O(n)
- **Vector Search:** Sub-linear (FAISS optimized)
- **Bottleneck:** LLM inference time

### Resource Usage

- **Memory:** ~2GB (embeddings + FAISS + LLM)
- **Disk:** ~500MB (models + embeddings)
- **CPU:** Moderate (inference dependent on hardware)

## Extension Points

### Adding New Models
```python
# config.py
MODELS = {
    'qwen': 'qwen2.5:7b',
    'llama': 'llama3.1:8b',
    'deepseek': 'deepseek-r1-distill-llama-8b',
    'custom': 'custom-model:tag'  # Add here
}
```

### Custom Embeddings
```python
# config.py
EMBEDDING_MODEL = 'custom-embedding-model'
```

### Modify Prompt Template
```python
# rag_engine.py - _build_system_prompt()
# Edit prompt string to change behavior
```

### Add Validation Rules
```python
# yaml_generator.py - validate_parameters()
# Add custom validation logic
```

## Testing

### Structure Test
```bash
python test_pipeline.py
```

**Checks:**
- All modules import successfully
- Configuration is valid
- Schema loader works
- YAML generator works

### Integration Test
```bash
# Use provided example
python main.py \
  -snippets example_snippets/snippet_1.txt \
  -dir test_outputs \
  -schemas example_schemas
```

## Troubleshooting Guide

### Common Issues

**"Ollama not available"**
```bash
ollama serve  # Start Ollama service
```

**"Model not found"**
```bash
ollama pull qwen2.5:7b
```

**"No schemas found"**
```bash
ls schemas/*.json  # Verify schemas exist
```

**"YAML parsing failed"**
- LLM output may be malformed
- Try running again (non-deterministic)
- Try different model

**Low novel parameter count**
- Increase top-k: `-k 7`
- Adjust prompt for more creativity
- Use reasoning-focused model (deepseek)

## Best Practices

### For Users

1. **Start with qwen** - Fastest, good quality
2. **Use high top-k for complex snippets** - More schema context
3. **Compare models** - Different models find different parameters
4. **Review outputs** - Validate extracted parameters
5. **Provide clear snippets** - Well-formatted input = better output

### For Developers

1. **Keep system prompt focused** - Clear instructions
2. **Monitor token usage** - Balance context vs cost
3. **Validate outputs** - Check YAML structure
4. **Log errors properly** - Informative error messages
5. **Test with edge cases** - Empty snippets, malformed schemas

## Future Enhancements

### Potential Features
- [ ] Caching for vector store (faster reruns)
- [ ] Parallel model execution
- [ ] Web interface for visualization
- [ ] Parameter deduplication across models
- [ ] Confidence scoring for parameters
- [ ] Fine-tuned embeddings for RISC-V
- [ ] Schema relationship awareness
- [ ] Output format options (JSON, CSV)

### Performance Optimizations
- [ ] Batch LLM inference
- [ ] GPU acceleration for embeddings
- [ ] Incremental vector store updates
- [ ] Response streaming

## License & Credits

**Pipeline Author:** Implementation based on requirements specification  
**Models:** Ollama (qwen, llama, deepseek)  
**Frameworks:** LangChain, FAISS, sentence-transformers  
**Purpose:** RISC-V architectural parameter extraction research

## Getting Help

### Documentation
1. **Quick Start:** See `QUICKSTART.md`
2. **Full Docs:** See `README.md`
3. **Architecture:** See `ARCHITECTURE.md`
4. **Example:** See `USAGE_EXAMPLE.md`

### Support
- Check console output for error details
- Verify all prerequisites installed
- Review example usage in `example.sh`
- Run `python test_pipeline.py` to validate setup

## Summary

This pipeline provides a **production-ready, extensible system** for extracting architectural parameters from RISC-V specifications using state-of-the-art RAG techniques. It balances **simplicity, performance, and functionality** while maintaining clean, maintainable code.

**Ready to use in 5 minutes. Powerful enough for production.**