import json
from pathlib import Path
from typing import Dict, List, Any

class SchemaLoader:
    def __init__(self, schema_dir: Path):
        self.schema_dir = Path(schema_dir)
        self.schemas = {}
        
    def load_all_schemas(self) -> Dict[str, Any]:
        schema_files = list(self.schema_dir.glob('*.json'))
        for schema_file in schema_files:
            if schema_file.name == 'json-schema-draft-07.json':
                continue
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                    self.schemas[schema_file.stem] = {
                        'filename': schema_file.name,
                        'data': schema_data,
                        'metadata': self._extract_metadata(schema_data)
                    }
            except Exception as e:
                print(f"Warning: Failed to load {schema_file.name}: {e}")
        return self.schemas
    
    def _extract_metadata(self, schema: Dict) -> Dict[str, Any]:
        metadata = {
            'description': schema.get('description', ''),
            'properties': {},
            'definitions': {}
        }
        
        if 'properties' in schema:
            metadata['properties'] = self._extract_properties(schema['properties'])
        
        if '$defs' in schema:
            for def_name, def_data in schema['$defs'].items():
                metadata['definitions'][def_name] = {
                    'description': def_data.get('description', ''),
                    'type': def_data.get('type', ''),
                    'properties': self._extract_properties(def_data.get('properties', {}))
                }
        
        return metadata
    
    def _extract_properties(self, properties: Dict) -> Dict[str, Any]:
        extracted = {}
        for prop_name, prop_data in properties.items():
            if isinstance(prop_data, dict):
                extracted[prop_name] = {
                    'type': prop_data.get('type', ''),
                    'description': prop_data.get('description', ''),
                    'enum': prop_data.get('enum', []),
                    'default': prop_data.get('default', None)
                }
        return extracted
    
    def get_schema_names(self) -> List[str]:
        return list(self.schemas.keys())
    
    def get_schema_by_name(self, name: str) -> Dict[str, Any]:
        return self.schemas.get(name, {})