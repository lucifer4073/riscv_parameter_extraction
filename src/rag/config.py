from pathlib import Path

MODELS = {
    'qwen': 'qwen2.5:7b',
    'llama': 'llama3.1:8b',
    'deepseek': 'deepseek-r1:8b'
}

DEFAULT_MODEL = 'qwen'
DEFAULT_TOP_K = 5
DEFAULT_VERSION = 1
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

YAML_FIELDS = [
    'name', 'description', 'type', 'unit', 'source_quote', 
    'rationale', 'constraints', 'scope', 'possible_values', 
    'value', 'properties', 'interface_type', 'additional_requirements'
]