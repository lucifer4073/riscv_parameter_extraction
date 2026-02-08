#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from rag.config import MODELS, DEFAULT_MODEL, DEFAULT_TOP_K
from rag.schema_loader import SchemaLoader
from rag.vector_store import VectorStore
from rag.llm_processor import LLMProcessor
from rag.rag_engine import RAGEngine
from rag.yaml_generator import YAMLGenerator
from rag.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(
        description='RISC-V Parameter Extraction RAG Pipeline'
    )
    parser.add_argument(
        '-snippets',
        required=True,
        help='Path to snippet file (.txt) or directory containing snippets'
    )
    parser.add_argument(
        '-dir',
        required=True,
        help='Directory to save output YAML files'
    )
    parser.add_argument(
        '-schemas',
        required=True,
        help='Directory containing JSON schema files'
    )
    parser.add_argument(
        '-model',
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f'Model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=DEFAULT_TOP_K,
        help=f'Number of schemas to retrieve (default: {DEFAULT_TOP_K})'
    )
    
    args = parser.parse_args()
    
    snippet_path = Path(args.snippets)
    output_dir = Path(args.dir)
    schema_dir = Path(args.schemas)
    model_key = args.model
    top_k = args.k
    
    if not schema_dir.exists():
        print(f"Error: Schema directory not found: {schema_dir}")
        sys.exit(1)
    
    if not snippet_path.exists():
        print(f"Error: Snippet path not found: {snippet_path}")
        sys.exit(1)
    
    model_name = MODELS[model_key]
    
    print(f"\n{'='*60}")
    print(f"RISC-V Parameter Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Top-K Schemas: {top_k}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    print("Loading schemas...")
    schema_loader = SchemaLoader(schema_dir)
    schemas = schema_loader.load_all_schemas()
    print(f"Loaded {len(schemas)} schemas")
    
    print("Building vector store...")
    vector_store = VectorStore()
    vector_store.build_from_schemas(schemas)
    print("Vector store ready")
    
    print(f"Initializing LLM ({model_name})...")
    llm_processor = LLMProcessor(model_name)
    if not llm_processor.check_availability():
        print("Error: Ollama not available. Please ensure Ollama is running.")
        sys.exit(1)
    print("LLM ready")
    
    rag_engine = RAGEngine(vector_store, llm_processor)
    yaml_generator = YAMLGenerator(output_dir)
    evaluator = Evaluator(schema_loader)
    
    snippet_files = []
    if snippet_path.is_file():
        snippet_files = [snippet_path]
    elif snippet_path.is_dir():
        snippet_files = list(snippet_path.glob('*.txt'))
    
    if not snippet_files:
        print(f"Error: No .txt files found in {snippet_path}")
        sys.exit(1)
    
    print(f"\nProcessing {len(snippet_files)} snippet(s)...\n")
    
    for snippet_file in snippet_files:
        print(f"Processing: {snippet_file.name}")
        
        try:
            with open(snippet_file, 'r') as f:
                snippet_text = f.read()
            
            result = rag_engine.process_snippet(snippet_text, top_k=top_k)
            
            print(f"  Primary Category: {result['primary_category']}")
            print(f"  Retrieved Schemas: {', '.join(result['retrieved_schemas'])}")
            
            parsed_params = yaml_generator.parse_llm_response(result['response'])
            
            if not parsed_params:
                print(f"  Warning: No parameters extracted")
            
            snippet_name = snippet_file.stem
            output_path = yaml_generator.save_yaml(
                parsed_params, 
                model_key, 
                snippet_name
            )
            
            print(f"  Saved: {output_path.name}")
            
            metrics = evaluator.evaluate(parsed_params)
            evaluator.print_metrics(metrics, model_name, snippet_file.name)
            
        except Exception as e:
            print(f"  Error processing {snippet_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully")
    print(f"Output files saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()