# Pipeline Architecture

## Overview

The RAG pipeline extracts architectural parameters from RISC-V specification snippets using retrieval-augmented generation with Ollama models.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │ RISC-V       │         │ JSON Schema     │              │
│  │ Snippets     │         │ Definitions     │              │
│  │ (.txt)       │         │ (22 schemas)    │              │
│  └──────┬───────┘         └────────┬────────┘              │
└─────────┼──────────────────────────┼──────────────────────┘
          │                          │
          │                          ▼
          │                  ┌───────────────┐
          │                  │ Schema Loader │
          │                  │ (JSON Parser) │
          │                  └───────┬───────┘
          │                          │
          │                          ▼
          │                  ┌───────────────────┐
          │                  │  Vector Store     │
          │                  │  (FAISS)          │
          │                  │  - Embeddings     │
          │                  │  - Metadata       │
          │                  └───────┬───────────┘
          │                          │
          ▼                          │
  ┌──────────────┐                  │
  │ Snippet Text │                  │
  └──────┬───────┘                  │
         │                          │
         │         ┌────────────────┘
         │         │
         ▼         ▼
  ┌──────────────────────────┐
  │    RAG Engine            │
  │  ┌────────────────────┐  │
  │  │ Similarity Search  │  │
  │  │ (Top-K Retrieval)  │  │
  │  └─────────┬──────────┘  │
  │            │              │
  │  ┌─────────▼──────────┐  │
  │  │ Category Detection │  │
  │  └─────────┬──────────┘  │
  │            │              │
  │  ┌─────────▼──────────┐  │
  │  │ Prompt Generation  │  │
  │  │ - System Prompt    │  │
  │  │ - User Prompt      │  │
  │  └─────────┬──────────┘  │
  └────────────┼─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │    LLM Processor         │
  │  (Ollama)                │
  │  - qwen2.5:7b            │
  │  - llama3.1:8b           │
  │  - deepseek-r1-distill   │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │   YAML Generator         │
  │  - Parse Response        │
  │  - Validate Structure    │
  │  - Auto-version          │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │      Evaluator           │
  │  - Count Parameters      │
  │  - Identify Novel Params │
  │  - Compute Metrics       │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │    Output Layer          │
  │  ┌────────────────────┐  │
  │  │ YAML Files         │  │
  │  │ {model}_{name}_v1  │  │
  │  └────────────────────┘  │
  │  ┌────────────────────┐  │
  │  │ Metrics Report     │  │
  │  │ (Console Output)   │  │
  │  └────────────────────┘  │
  └──────────────────────────┘
```

## Component Details

### 1. Schema Loader (`schema_loader.py`)
**Purpose**: Load and parse JSON schema files

**Key Methods**:
- `load_all_schemas()`: Load all JSON schemas from directory
- `_extract_metadata()`: Extract descriptions, properties, definitions
- `_extract_properties()`: Parse property definitions

**Output**: Dictionary of schemas with metadata

### 2. Vector Store (`vector_store.py`)
**Purpose**: Create searchable embeddings of schemas

**Technology**: 
- FAISS for vector similarity search
- HuggingFace sentence-transformers for embeddings
- Model: all-MiniLM-L6-v2 (384 dimensions)

**Key Methods**:
- `build_from_schemas()`: Create embeddings from schemas
- `retrieve_top_k()`: Retrieve k most similar schemas

**Retrieval Logic**:
- Convert schemas to text representations
- Embed using sentence-transformers
- Store in FAISS index
- Perform similarity search on snippet text

### 3. LLM Processor (`llm_processor.py`)
**Purpose**: Interface with Ollama models

**Supported Models**:
- qwen2.5:7b (default)
- llama3.1:8b
- deepseek-r1-distill-llama-8b

**Configuration**:
- Temperature: 0.7 (balanced creativity)
- Top-p: 0.9 (nucleus sampling)

### 4. RAG Engine (`rag_engine.py`)
**Purpose**: Core RAG logic and prompt engineering

**Workflow**:
1. Retrieve top-k relevant schemas
2. Detect primary category (highest scoring schema)
3. Build system prompt with schema context
4. Generate user prompt with snippet
5. Call LLM and return response

**System Prompt Strategy**:
- Inject retrieved schema definitions
- Specify YAML output format
- Emphasize novel parameter discovery
- Require source quotes for all parameters

### 5. YAML Generator (`yaml_generator.py`)
**Purpose**: Parse LLM output and generate YAML files

**Features**:
- Parse YAML from LLM response
- Clean markdown code blocks
- Auto-increment version numbers
- Validate required fields

**Validation**:
- Required fields: name, description, type, source_quote, rationale, scope
- Optional fields: unit, constraints, possible_values, etc.

### 6. Evaluator (`evaluator.py`)
**Purpose**: Compute extraction metrics

**Metrics**:
- **Total Parameters**: Count all extracted items
- **Unique Parameters**: Count distinct parameter names
- **Novel Parameters**: Parameters not in schemas (fuzzy matching)
- **Duplicates**: Repeated parameters

**Novel Detection Algorithm**:
```python
for param in extracted:
    normalized_param = normalize(param.name)
    is_novel = True
    for schema_param in all_schema_params:
        normalized_schema = normalize(schema_param)
        if match(normalized_param, normalized_schema):
            is_novel = False
            break
    if is_novel:
        novel_count += 1
```

## Data Flow

### Single Snippet Processing

```
1. Load snippet text from .txt file
   ↓
2. Embed snippet using sentence-transformers
   ↓
3. FAISS similarity search → Top-K schemas
   ↓
4. Primary category = schema with highest score
   ↓
5. Build system prompt:
   - Include all K schema definitions
   - Specify primary category context
   - Define YAML output format
   ↓
6. Build user prompt:
   - Include snippet text
   - Request extraction
   ↓
7. Call Ollama model with prompts
   ↓
8. Parse YAML from LLM response
   ↓
9. Validate parameter structure
   ↓
10. Auto-version and save YAML file
   ↓
11. Compute and display metrics
```

### Batch Processing (Multiple Snippets)

```
For each snippet in directory:
    Process as single snippet (steps 1-11)
    Continue to next snippet
```

### Multi-Model Processing

```
For each model in [qwen, llama, deepseek]:
    For each snippet:
        Process snippet with current model
        Generate separate YAML: {model}_{snippet}_v{N}.yaml
```

## Configuration

### Embedding Model Selection
- **Model**: all-MiniLM-L6-v2
- **Rationale**: 
  - Good balance of speed and quality
  - 384-dimensional embeddings
  - Well-suited for semantic similarity

### Top-K Selection
- **Default**: k=5
- **Rationale**: Provides diverse schema context without overwhelming LLM

### LLM Parameters
- **Temperature**: 0.7 (allows creativity for novel params)
- **Top-p**: 0.9 (maintains quality while exploring)

## Extension Points

### Adding New Models
1. Add to `MODELS` dict in `config.py`
2. Ensure model is pulled in Ollama
3. No code changes needed

### Custom Embeddings
1. Change `EMBEDDING_MODEL` in `config.py`
2. Must be compatible with HuggingFace sentence-transformers

### Custom Prompt Templates
1. Modify `_build_system_prompt()` in `rag_engine.py`
2. Adjust emphasis on novel vs schema-defined parameters

### Output Format Changes
1. Modify YAML structure in `yaml_generator.py`
2. Update validation in `validate_parameters()`

## Performance Considerations

### First Run
- Downloads embedding model (~80MB)
- Builds vector store from schemas
- Time: ~30-60 seconds

### Subsequent Runs
- Embedding model cached locally
- Vector store rebuilt (fast: <5 seconds)
- LLM inference: 5-30 seconds per snippet

### Scalability
- **Schemas**: Linear with number of schemas
- **Snippets**: Linear with number of snippets
- **Vector Search**: Sub-linear (FAISS optimized)
- **Bottleneck**: LLM inference time

## Error Handling

### Ollama Unavailable
- Check with `llm_processor.check_availability()`
- Exit with clear error message

### YAML Parsing Failure
- Log warning
- Return empty list
- Save empty YAML with warning

### Schema Loading Error
- Skip problematic schema
- Log warning
- Continue with remaining schemas

### No Parameters Extracted
- Log warning
- Save empty YAML file
- Display zero metrics