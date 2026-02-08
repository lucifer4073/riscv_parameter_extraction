#!/usr/bin/env python3
import sys
from pathlib import Path

def test_imports():
    print("Testing imports...")
    try:
        import config
        import schema_loader
        import vector_store
        import llm_processor
        import rag_engine
        import yaml_generator
        import evaluator
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    print("\nTesting configuration...")
    try:
        from config import MODELS, DEFAULT_MODEL, EMBEDDING_MODEL
        print(f"  Models: {', '.join(MODELS.keys())}")
        print(f"  Default: {DEFAULT_MODEL}")
        print(f"  Embedding: {EMBEDDING_MODEL}")
        print("✓ Configuration valid")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_schema_loader():
    print("\nTesting schema loader...")
    try:
        from schema_loader import SchemaLoader
        loader = SchemaLoader(Path('.'))
        print("✓ SchemaLoader initialized")
        return True
    except Exception as e:
        print(f"✗ SchemaLoader test failed: {e}")
        return False

def test_yaml_generator():
    print("\nTesting YAML generator...")
    try:
        from yaml_generator import YAMLGenerator
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = YAMLGenerator(Path(tmpdir))
            test_data = [
                {
                    'name': 'test_param',
                    'description': 'Test parameter',
                    'type': 'Integer',
                    'source_quote': 'test quote',
                    'rationale': 'test rationale',
                    'scope': 'test scope'
                }
            ]
            output = generator.save_yaml(test_data, 'test', 'snippet')
            print(f"  Generated: {output.name}")
            print("✓ YAML generator works")
            return True
    except Exception as e:
        print(f"✗ YAML generator test failed: {e}")
        return False

def main():
    print("="*60)
    print("RAG Pipeline Structure Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_config,
        test_schema_loader,
        test_yaml_generator
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed! Pipeline structure is valid.")
        return 0
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())