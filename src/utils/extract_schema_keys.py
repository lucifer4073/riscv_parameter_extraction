#!/usr/bin/env python3
"""
Extract unique parameter names from JSON schema files.
This module is used to compute novelty scores for YAML outputs.
"""

import json
from pathlib import Path
from typing import Set, List, Dict, Any
import sys


def extract_properties_from_object(obj: Any, collected_keys: Set[str]) -> None:
    """
    Recursively extract all keys from 'properties', 'patternProperties', 
    and object definitions in a JSON schema.
    
    Args:
        obj: The JSON object (dict, list, or other) to traverse
        collected_keys: Set to collect all unique property names
    """
    if isinstance(obj, dict):
        # Extract keys from 'properties' field
        if 'properties' in obj:
            collected_keys.update(obj['properties'].keys())
            # Recursively explore nested properties
            for value in obj['properties'].values():
                extract_properties_from_object(value, collected_keys)
        
        # Extract keys from 'patternProperties' field
        if 'patternProperties' in obj:
            for pattern_obj in obj['patternProperties'].values():
                extract_properties_from_object(pattern_obj, collected_keys)
        
        # Recursively traverse all other values in the dict
        for key, value in obj.items():
            if key not in ['properties', 'patternProperties']:
                extract_properties_from_object(value, collected_keys)
    
    elif isinstance(obj, list):
        # Recursively explore list items
        for item in obj:
            extract_properties_from_object(item, collected_keys)


def extract_keys_from_schema(schema_path: str) -> Set[str]:
    """
    Extract all unique parameter names from a single JSON schema file.
    
    Args:
        schema_path: Path to the JSON schema file
        
    Returns:
        Set of unique parameter names found in the schema
    """
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        keys = set()
        extract_properties_from_object(schema, keys)
        
        # Filter out common schema keywords that aren't actual parameter names
        schema_keywords = {
            '$schema', '$ref', '$defs', '$comment', 'type', 'description',
            'required', 'properties', 'items', 'enum', 'const', 'pattern',
            'minimum', 'maximum', 'oneOf', 'anyOf', 'allOf', 'if', 'then',
            'else', 'additionalProperties', 'patternProperties', 'format',
            'title', 'kind', '$source', 'minItems', 'maxItems'
        }
        
        # Remove schema keywords, keep only actual property names
        actual_keys = keys - schema_keywords
        
        return actual_keys
    
    except Exception as e:
        print(f"Warning: Could not parse {schema_path}: {e}")
        return set()


def extract_all_schema_keys(schema_directory: str, verbose: bool = True) -> Set[str]:
    """
    Extract all unique parameter names from all JSON schema files in a directory.
    
    Args:
        schema_directory: Path to directory containing JSON schema files
        verbose: Whether to print progress information
        
    Returns:
        Set of all unique parameter names across all schemas
    """
    schema_dir = Path(schema_directory)
    
    if not schema_dir.exists():
        raise ValueError(f"Schema directory does not exist: {schema_directory}")
    
    all_keys = set()
    schema_files = list(schema_dir.glob("*.json"))
    
    if not schema_files:
        raise ValueError(f"No JSON schema files found in: {schema_directory}")
    
    if verbose:
        print(f"\nExtracting standard parameters from {len(schema_files)} schema files:")
    
    for schema_file in schema_files:
        if verbose:
            print(f"  - {schema_file.name}")
        
        file_keys = extract_keys_from_schema(str(schema_file))
        all_keys.update(file_keys)
        
        if verbose:
            print(f"    Found {len(file_keys)} unique keys")
    
    if verbose:
        print(f"\nTotal unique standard parameters: {len(all_keys)}")
        print(f"Standard parameters: {sorted(all_keys)}\n")
    
    return all_keys


def compute_novelty_score(yaml_keys: Set[str], schema_keys: Set[str]) -> Dict[str, Any]:
    """
    Compute novelty metrics for a set of YAML keys against standard schema keys.
    
    Args:
        yaml_keys: Set of keys extracted from a YAML file
        schema_keys: Set of standard keys from schema files
        
    Returns:
        Dictionary containing:
            - novelty_score: Number of novel parameters (not in schema)
            - novelty_ratio: Ratio of novel to total parameters
            - novel_params: List of novel parameter names
            - standard_params: List of parameters that match schema
    """
    novel = yaml_keys - schema_keys
    standard = yaml_keys & schema_keys
    
    novelty_score = len(novel)
    total = len(yaml_keys)
    novelty_ratio = novelty_score / total if total > 0 else 0
    
    return {
        'novelty_score': novelty_score,
        'novelty_ratio': novelty_ratio,
        'novel_params': sorted(novel),
        'standard_params': sorted(standard),
        'total_yaml_params': total,
        'matched_standard_params': len(standard)
    }


def save_schema_keys(schema_keys: Set[str], output_path: str) -> None:
    """
    Save extracted schema keys to a file for reference.
    
    Args:
        schema_keys: Set of schema parameter names
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        f.write("# Standard Schema Parameters\n")
        f.write(f"# Total: {len(schema_keys)}\n\n")
        for key in sorted(schema_keys):
            f.write(f"{key}\n")
    
    print(f"Schema keys saved to: {output_path}")


def main():
    """
    Command-line interface for schema key extraction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract unique parameter names from JSON schema files'
    )
    parser.add_argument('schema_directory', type=str, 
                       help='Directory containing JSON schema files')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output file to save extracted keys')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        schema_keys = extract_all_schema_keys(args.schema_directory, 
                                              verbose=not args.quiet)
        
        if args.output:
            save_schema_keys(schema_keys, args.output)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())