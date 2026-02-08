import yaml
import re
from pathlib import Path
from typing import Dict, Any, List

class YAMLGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        cleaned_response = self._clean_response(response)
        
        try:
            parsed = yaml.safe_load(cleaned_response)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                return []
        except yaml.YAMLError as e:
            print(f"Warning: YAML parsing failed: {e}")
            return []
    
    def _clean_response(self, response: str) -> str:
        response = response.strip()
        
        if '```yaml' in response:
            match = re.search(r'```yaml\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif '```' in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        return response.strip()
    
    def save_yaml(self, data: List[Dict[str, Any]], model_name: str, 
                  snippet_name: str, version: int = 1) -> Path:
        if not data:
            print(f"Warning: No parameters extracted for {snippet_name}")
            data = []
        
        output_filename = f"{model_name}_{snippet_name}_v{version}.yaml"
        output_path = self.output_dir / output_filename
        
        output_path = self._auto_version(output_path, model_name, snippet_name)
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, width=120)
        
        return output_path
    
    def _auto_version(self, output_path: Path, model_name: str, snippet_name: str) -> Path:
        version = 1
        while output_path.exists():
            version += 1
            output_filename = f"{model_name}_{snippet_name}_v{version}.yaml"
            output_path = self.output_dir / output_filename
        return output_path
    
    def validate_parameters(self, data: List[Dict[str, Any]]) -> bool:
        required_fields = {'name', 'description', 'type', 'source_quote', 'rationale', 'scope'}
        
        for item in data:
            if not isinstance(item, dict):
                return False
            if not required_fields.issubset(item.keys()):
                missing = required_fields - item.keys()
                print(f"Warning: Parameter missing required fields: {missing}")
        
        return True