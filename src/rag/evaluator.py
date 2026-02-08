from typing import List, Dict, Any, Set

class Evaluator:
    def __init__(self, schema_loader):
        self.schema_loader = schema_loader
        self.all_schema_params = self._extract_all_schema_params()
        
    def _extract_all_schema_params(self) -> Set[str]:
        all_params = set()
        for schema_name, schema_info in self.schema_loader.schemas.items():
            metadata = schema_info['metadata']
            if 'properties' in metadata:
                all_params.update(metadata['properties'].keys())
            if 'definitions' in metadata:
                for def_name, def_info in metadata['definitions'].items():
                    if 'properties' in def_info:
                        all_params.update(def_info['properties'].keys())
        return all_params
    
    def evaluate(self, extracted_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_params = len(extracted_params)
        
        param_names = [p.get('name', '') for p in extracted_params]
        unique_params = len(set(param_names))
        
        novel_params = self._count_novel_params(param_names)
        
        metrics = {
            'total_parameters': total_params,
            'unique_parameters': unique_params,
            'novel_parameters': novel_params,
            'parameter_names': param_names,
            'duplicates': total_params - unique_params
        }
        
        return metrics
    
    def _count_novel_params(self, param_names: List[str]) -> int:
        novel_count = 0
        for param in param_names:
            param_normalized = param.lower().replace('_', '').replace('-', '')
            is_novel = True
            for schema_param in self.all_schema_params:
                schema_normalized = schema_param.lower().replace('_', '').replace('-', '')
                if param_normalized == schema_normalized or param_normalized in schema_normalized:
                    is_novel = False
                    break
            if is_novel:
                novel_count += 1
        return novel_count
    
    def print_metrics(self, metrics: Dict[str, Any], model_name: str, snippet_name: str):
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Snippet: {snippet_name}")
        print(f"{'='*60}")
        print(f"├── Total Parameters: {metrics['total_parameters']}")
        print(f"├── Unique Parameters: {metrics['unique_parameters']}")
        print(f"├── Novel Parameters: {metrics['novel_parameters']}")
        print(f"└── Duplicates: {metrics['duplicates']}")
        if metrics['parameter_names']:
            print(f"\nExtracted Parameters:")
            for i, name in enumerate(metrics['parameter_names'], 1):
                print(f"  {i}. {name}")
        print(f"{'='*60}\n")