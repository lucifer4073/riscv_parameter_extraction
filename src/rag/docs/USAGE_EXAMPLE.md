# Usage Example with Real Data

This guide walks through processing the provided example snippet.

## Input Data

### Snippet (snippet_1.txt)
```
"Conventional" R/W accessibility of CSRs according to address mapping 
The standard RISC-V ISA sets aside a 12-bit encoding space (csr[11:0]) for up to 4,096 CSRs. By convention, the upper 4 bits of the CSR address (csr[11:8]) are used to encode the read and write accessibility of the CSRs according to privilege level as shown in Table 1. The top two bits (csr[11:10]) indicate whether the register is read/write (00,01, or 10) or read-only (11). The next two bits (csr[9:8]) encode the lowest privilege level that can access the CSR.
```

### Expected Output Structure
Based on the provided example (`claude_S1-4.5-snippet_1.yaml`), parameters should include:

**Required Fields:**
- `name`: Parameter identifier
- `description`: What the parameter represents  
- `type`: Data type (Integer, String, Enum, etc.)
- `source_quote`: Direct quote from snippet
- `rationale`: Why this is architecturally significant
- `scope`: Application scope

**Optional Fields:**
- `unit`: Measurement unit (bytes, bits, etc.)
- `constraints`: List of constraints
- `possible_values`: Valid values for enums
- `value`: Specific value if mentioned
- `properties`: Sub-properties
- `additional_requirements`: Extra notes

## Running the Example

### Step 1: Setup Directory Structure
```bash
mkdir -p example_run/snippets example_run/schemas example_run/outputs
```

### Step 2: Place Files
```bash
# Copy snippet
cp snippet_1.txt example_run/snippets/

# Copy schema files
cp schemas/*.json example_run/schemas/

# Copy pipeline files
cp *.py example_run/
```

### Step 3: Run with Default Model (qwen)
```bash
cd example_run
python main.py \
  -snippets snippets/snippet_1.txt \
  -dir outputs \
  -schemas schemas
```

### Expected Output (Console)

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

Pipeline completed successfully
Output files saved to: outputs
============================================================
```

## Output File Analysis

### Generated YAML (qwen_snippet_1_v1.yaml)

Expected structure similar to:

```yaml
- name: csr_encoding_space_size
  description: The bit width of the CSR address encoding space
  type: Integer
  unit: bits
  value: 12
  source_quote: "The standard RISC-V ISA sets aside a 12-bit encoding space (csr[11:0])"
  rationale: Defines the fundamental addressing capacity for CSRs in the RISC-V architecture
  scope: ISA-wide specification
  constraints:
    - Fixed at 12 bits in standard RISC-V ISA
  
- name: csr_total_addressable_count
  description: Maximum number of CSRs that can be addressed
  type: Integer
  value: 4096
  source_quote: "12-bit encoding space (csr[11:0]) for up to 4,096 CSRs"
  rationale: Direct consequence of 12-bit encoding (2^12 = 4096 possible addresses)
  scope: ISA-wide specification
  constraints:
    - Maximum of 4096 CSRs addressable

- name: csr_read_write_accessibility_encoding
  description: Encoding scheme for CSR read/write permissions using top 2 bits
  type: Enum
  possible_values:
    - "00: read/write"
    - "01: read/write"
    - "10: read/write"
    - "11: read-only"
  source_quote: "The top two bits (csr[11:10]) indicate whether the register is read/write (00,01, or 10) or read-only (11)"
  rationale: Enables hardware to enforce access permissions based on CSR address alone
  scope: Per CSR instance
  constraints:
    - Encoded in bits [11:10] of CSR address
    - Values 00, 01, 10 indicate read/write
    - Value 11 indicates read-only

- name: csr_privilege_level_encoding
  description: Encoding of minimum privilege level required to access CSR
  type: Bitfield
  source_quote: "The next two bits (csr[9:8]) encode the lowest privilege level that can access the CSR"
  rationale: Provides privilege-based access control encoded directly in CSR address
  scope: Per CSR instance
  constraints:
    - Encoded in bits [9:8] of CSR address
    - Represents minimum privilege level for access

- name: csr_address_upper_bits_function
  description: Functional interpretation of upper 4 bits of CSR address
  type: Composite_Encoding
  source_quote: "the upper 4 bits of the CSR address (csr[11:8]) are used to encode the read and write accessibility of the CSRs according to privilege level"
  rationale: Demonstrates address space partitioning strategy in RISC-V
  scope: ISA-wide convention
  properties:
    - bits_11_10: read/write accessibility
    - bits_9_8: privilege level encoding

- name: csr_address_mapping_convention
  description: Conventional approach to encoding CSR properties in address bits
  type: Architectural_Convention
  source_quote: "By convention, the upper 4 bits of the CSR address (csr[11:8]) are used to encode the read and write accessibility"
  rationale: Novel parameter - represents design philosophy of encoding metadata in addresses
  scope: ISA-wide design principle
  additional_requirements: This is a convention, suggesting implementations could deviate but should maintain compatibility
```

## Comparing Models

### Run All Three Models
```bash
for model in qwen llama deepseek; do
    echo "Running with $model..."
    python main.py \
      -snippets snippets/snippet_1.txt \
      -dir outputs \
      -schemas schemas \
      -model $model
done
```

### Expected Outputs
```
outputs/
├── qwen_snippet_1_v1.yaml       # Qwen extraction
├── llama_snippet_1_v1.yaml      # Llama extraction
└── deepseek_snippet_1_v1.yaml   # DeepSeek extraction
```

### Comparison Metrics

| Model | Total Params | Unique Params | Novel Params |
|-------|-------------|---------------|--------------|
| Qwen  | 6-8         | 6-8           | 3-5          |
| Llama | 5-7         | 5-7           | 2-4          |
| DeepSeek | 7-9      | 7-9           | 4-6          |

*Note: Exact numbers vary due to LLM non-determinism*

## Evaluation Criteria

### What Makes a Good Parameter?

**Schema-Defined Parameters:**
- Matches property names in retrieved schemas
- Has clear architectural meaning
- Well-documented with source quote

**Novel Parameters:**
- NOT explicitly in any schema
- Architecturally significant (not trivial)
- Derives from specification text
- Non-repetitive (doesn't duplicate existing params)

### Example Novel Parameters from Snippet

✓ **Good Novel Parameters:**
- `csr_address_mapping_convention` - Design philosophy
- `csr_address_upper_bits_function` - Composite encoding
- `csr_total_addressable_count` - Derived from encoding space

✗ **Poor Novel Parameters:**
- `csr_address` - Too vague, already in schemas
- `register_type` - Generic, not specific enough
- `bit_encoding` - Duplicates more specific parameters

## Advanced Usage

### Custom Schema Retrieval
```bash
# Retrieve only top 3 schemas
python main.py -snippets snippets/snippet_1.txt -dir outputs -schemas schemas -k 3

# More schemas for complex snippets
python main.py -snippets snippets/complex_snippet.txt -dir outputs -schemas schemas -k 10
```

### Batch Processing Different Snippet Types
```bash
# Process CSR-related snippets
python main.py -snippets snippets/csr_*.txt -dir outputs/csr -schemas schemas

# Process instruction-related snippets  
python main.py -snippets snippets/inst_*.txt -dir outputs/inst -schemas schemas
```

## Troubleshooting This Example

### Issue: "No parameters extracted"
**Solution:** Check if Ollama model is loaded
```bash
ollama list | grep qwen2.5
ollama pull qwen2.5:7b  # If not found
```

### Issue: "Schema directory not found"
**Solution:** Verify schema path
```bash
ls schemas/*.json | head -5
python main.py -schemas /absolute/path/to/schemas ...
```

### Issue: "YAML parsing failed"
**Solution:** LLM output may be malformed
- Check console for raw LLM response
- Try running again (non-deterministic)
- Try different model: `-model llama`

### Issue: Low novel parameter count
**Solution:** Adjust retrieval or prompt
- Increase top-k: `-k 7`
- Edit system prompt in `rag_engine.py` to emphasize novelty
- Use model with better reasoning: `-model deepseek`

## Next Steps

1. **Validate outputs**: Compare against `claude_S1-4.5-snippet_1.yaml`
2. **Test other models**: Compare extraction quality
3. **Process more snippets**: Build parameter database
4. **Analyze patterns**: Look for consistent novel parameters
5. **Refine prompts**: Improve extraction quality based on results