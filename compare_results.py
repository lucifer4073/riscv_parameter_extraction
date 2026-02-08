#!/usr/bin/env python3
"""
Generate comparison visualizations from YAML analysis results.

This script reads the CSV output from yaml_analyzer.py and creates
comparison charts and summary statistics.
"""

import csv
import sys
from pathlib import Path


def load_results(csv_file: str) -> list:
    """Load results from CSV file."""
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for field in ['parameter_sets', 'unique_properties', 'total_fields',
                         'unique_param_names_count', 'params_with_properties',
                         'params_with_description', 'params_with_type',
                         'params_with_source_quote', 'max_nesting_depth',
                         'unique_types_count']:
                if row.get(field):
                    row[field] = int(row[field])
            
            if row.get('avg_properties_per_param'):
                row['avg_properties_per_param'] = float(row['avg_properties_per_param'])
            
            results.append(row)
    return results


def print_comparison_table(results: list):
    """Print a formatted comparison table."""
    print("\n" + "="*120)
    print("MODEL COMPARISON TABLE")
    print("="*120)
    
    # Header
    print(f"{'Model ID':<20} {'Coverage':<10} {'Quality':<10} {'w/Props':<10} "
          f"{'w/Desc':<10} {'w/Type':<10} {'w/Quote':<10} {'Avg Props':<10}")
    print("-"*120)
    
    # Sort by quality metric (unique_properties)
    sorted_results = sorted(results, key=lambda x: x.get('unique_properties', 0), reverse=True)
    
    for result in sorted_results:
        model_id = result.get('model_id', 'N/A')[:18]
        coverage = result.get('parameter_sets', 0)
        quality = result.get('unique_properties', 0)
        with_props = result.get('params_with_properties', 0)
        with_desc = result.get('params_with_description', 0)
        with_type = result.get('params_with_type', 0)
        with_quote = result.get('params_with_source_quote', 0)
        avg_props = result.get('avg_properties_per_param', 0)
        
        print(f"{model_id:<20} {coverage:<10} {quality:<10} {with_props:<10} "
              f"{with_desc:<10} {with_type:<10} {with_quote:<10} {avg_props:<10.2f}")
    
    print("="*120)


def print_ranking(results: list):
    """Print rankings by different metrics."""
    print("\n" + "="*80)
    print("RANKINGS BY METRIC")
    print("="*80)
    
    metrics = [
        ('unique_properties', 'Quality (Unique Properties)'),
        ('parameter_sets', 'Coverage (Parameter Sets)'),
        ('params_with_description', 'Documentation Quality'),
        ('params_with_source_quote', 'Traceability'),
    ]
    
    for metric_key, metric_name in metrics:
        print(f"\n{metric_name}:")
        print("-" * 60)
        sorted_results = sorted(results, key=lambda x: x.get(metric_key, 0), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            model_id = result.get('model_id', 'N/A')
            value = result.get(metric_key, 0)
            print(f"  {i}. {model_id:<25} {value:>5}")


def print_statistics(results: list):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not results:
        print("No results to analyze")
        return
    
    metrics = {
        'parameter_sets': [],
        'unique_properties': [],
        'params_with_properties': [],
        'params_with_description': [],
        'params_with_source_quote': [],
        'avg_properties_per_param': []
    }
    
    for result in results:
        for metric in metrics:
            value = result.get(metric)
            if value is not None:
                metrics[metric].append(value)
    
    print(f"\nTotal Models Analyzed: {len(results)}")
    
    for metric_key, values in metrics.items():
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            metric_name = metric_key.replace('_', ' ').title()
            print(f"\n{metric_name}:")
            print(f"  Average: {avg:.2f}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Range: {max_val - min_val}")


def generate_markdown_report(results: list, output_file: str):
    """Generate a markdown report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# YAML Parameter Extraction Analysis Report\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total Models Analyzed**: {len(results)}\n")
        f.write(f"- **Date**: {Path(output_file).stat().st_mtime}\n\n")
        
        # Comparison Table
        f.write("## Model Comparison\n\n")
        f.write("| Model ID | Coverage | Quality | With Props | With Desc | With Type | With Quote | Avg Props |\n")
        f.write("|----------|----------|---------|------------|-----------|-----------|------------|----------|\n")
        
        sorted_results = sorted(results, key=lambda x: x.get('unique_properties', 0), reverse=True)
        for result in sorted_results:
            f.write(f"| {result.get('model_id', 'N/A')} | "
                   f"{result.get('parameter_sets', 0)} | "
                   f"{result.get('unique_properties', 0)} | "
                   f"{result.get('params_with_properties', 0)} | "
                   f"{result.get('params_with_description', 0)} | "
                   f"{result.get('params_with_type', 0)} | "
                   f"{result.get('params_with_source_quote', 0)} | "
                   f"{result.get('avg_properties_per_param', 0):.2f} |\n")
        
        f.write("\n## Metrics Explained\n\n")
        f.write("- **Coverage**: Total number of parameter sets extracted\n")
        f.write("- **Quality**: Total unique sub-parameters (properties) found\n")
        f.write("- **With Props**: Parameters that have a properties field\n")
        f.write("- **With Desc**: Parameters with descriptions\n")
        f.write("- **With Type**: Parameters with type information\n")
        f.write("- **With Quote**: Parameters with source quotes\n")
        f.write("- **Avg Props**: Average properties per parameter\n")
    
    print(f"\nMarkdown report saved to: {output_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate comparison visualizations from YAML analysis'
    )
    parser.add_argument(
        'csv_file',
        help='CSV file from yaml_analyzer.py'
    )
    parser.add_argument(
        '-m', '--markdown',
        help='Generate markdown report (specify output file)'
    )
    
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_results(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    if not results:
        print("No results found in CSV file")
        sys.exit(1)
    
    # Print analysis
    print_comparison_table(results)
    print_ranking(results)
    print_statistics(results)
    
    # Generate markdown if requested
    if args.markdown:
        generate_markdown_report(results, args.markdown)


if __name__ == '__main__':
    main()