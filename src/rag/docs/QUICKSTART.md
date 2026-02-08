# Quick Start Guide

Get started with the RISC-V Parameter Extraction Pipeline in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Install and Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull models (choose at least one)
ollama pull qwen2.5:7b      # Default, recommended
ollama pull llama3.1:8b     # Alternative
ollama run deepseek-r1:8b # Alternative
```

## Step 3: Prepare Your Files

### Directory Structure
```
your_project/
├── main.py              # Pipeline entry point
├── config.py            # Configuration
├── [other .py files]    # Pipeline modules
├── schemas/             # Your JSON schema files
│   ├── inst_schema.json
│   ├── csr_schema.json
│   └── [other schemas]
└── snippets/            # Your RISC-V snippets
    ├── snippet_1.txt
    └── snippet_2.txt
```

### Snippet Format
Create `.txt` files with RISC-V specification text:

```text
# snippet_1.txt
The standard RISC-V ISA sets aside a 12-bit encoding space
for up to 4,096 CSRs. The top two bits (csr[11:10]) indicate
whether the register is read/write or read-only...
```

## Step 4: Run the Pipeline

### Single Snippet
```bash
python main.py \
  -snippets snippets/snippet_1.txt \
  -dir outputs \
  -schemas schemas
```

### All Snippets in Directory
```bash
python main.py \
  -snippets snippets \
  -dir outputs \
  -schemas schemas
```

### Specific Model
```bash
python main.py \
  -snippets snippets/snippet_1.txt \
  -dir outputs \
  -schemas schemas \
  -model llama
```

## Step 5: Check Results

### Output Files
```bash
ls outputs/
# qwen_snippet_1_v1.yaml
# llama_snippet_1_v1.yaml
```

### Metrics (Console Output)
```
============================================================
Model: qwen2.5:7b | Snippet: snippet_1.txt
============================================================
├── Total Parameters: 8
├── Unique Parameters: 7
├── Novel Parameters: 4
└── Duplicates: 1

Extracted Parameters:
  1. csr_encoding_space
  2. csr_address_width
  3. csr_read_write_indicator
  ...
============================================================
```

## Common Commands

### Test All Models
```bash
for model in qwen llama deepseek; do
    python main.py -snippets snippets/snippet_1.txt -dir outputs -schemas schemas -model $model
done
```

### Batch Process
```bash
python main.py -snippets snippets -dir outputs -schemas schemas
```

### Custom Retrieval
```bash
python main.py -snippets snippets/snippet_1.txt -dir outputs -schemas schemas -k 3
```

## Troubleshooting

### "Ollama not available"
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull qwen2.5:7b
```

### "No schemas found"
```bash
# Check schema directory
ls schemas/*.json

# Ensure path is correct
python main.py -schemas /absolute/path/to/schemas ...
```

### YAML Parsing Errors
- The LLM response may not be valid YAML
- Try running again (LLMs are non-deterministic)
- Check the console output for the raw response

## Next Steps

- Read [README.md](README.md) for detailed documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [example.sh](example.sh) for more usage patterns
- Run `python test_pipeline.py` to verify installation

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt
ollama pull qwen2.5:7b

# 2. Verify installation
python test_pipeline.py

# 3. Process snippets
python main.py -snippets snippets -dir outputs -schemas schemas

# 4. Compare models
for model in qwen llama deepseek; do
    python main.py -snippets snippets/snippet_1.txt -dir outputs -schemas schemas -model $model
done

# 5. Review outputs
cat outputs/qwen_snippet_1_v1.yaml
```

## Tips

1. **First run is slower**: Downloads embedding model (~80MB)
2. **Use qwen for speed**: Fastest of the three models
3. **Increase top-k for complex snippets**: `-k 7` or `-k 10`
4. **Version auto-increments**: Safe to run multiple times
5. **Novel params are valuable**: Check the metrics for discovery rate

## Need Help?

- Check console output for detailed error messages
- Ensure all prerequisites are installed
- Verify file paths are correct
- See README.md for full documentation