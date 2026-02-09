#!/usr/bin/env python3
"""
Analyze YAML output snippets from different LLM models.
Compares Quality (unique parameters per item) and Quantity (total items).
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import argparse
import sys

from src.utils.extract_schema_keys import extract_all_schema_keys, compute_novelty_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def count_unique_keys(item: dict) -> int:
    """Count unique top-level keys in a YAML item."""
    return len(item.keys())

def get_all_unique_keys(items: List[dict]) -> set:
    """Get all unique keys across all items in a file."""
    all_keys = set()
    for item in items:
        all_keys.update(item.keys())
    return all_keys

def analyze_yaml_file(filepath: str) -> Tuple[int, float, int, set]:
    """
    Analyze a single YAML file.
    
    Returns:
        quantity: Total number of items
        avg_quality: Average number of unique parameters per item
        max_quality: Maximum unique parameters in any single item
        all_keys: Set of all unique keys found (for novelty calculation)
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    quantity = len(data)
    all_keys = get_all_unique_keys(data)
    
    key_counts = [count_unique_keys(item) for item in data]
    avg_quality = np.mean(key_counts) if key_counts else 0
    max_quality = max(key_counts) if key_counts else 0
    
    return quantity, avg_quality, max_quality, all_keys

def extract_model_info(filename: str) -> Tuple[str, str, str]:
    """
    Extract model name, version, and snippet number from filename.
    Expected format: {model}_S{stage}-{version}-snippet_{number}.yaml
    
    Returns:
        model_name: The base model name (claude, gemini, gpt, grok)
        version: Full version string (e.g., S1-4.5, S2-3-pro)
        snippet_num: Snippet number
    """
    stem = Path(filename).stem
    
    parts = stem.split('_')
    if len(parts) < 2:
        return stem, "unknown", "1"
    
    model_name = parts[0]
    
    if '-snippet_' in stem:
        version_part, snippet_num = stem.split('-snippet_')
        version = version_part.replace(f"{model_name}_", "")
    else:
        version = '_'.join(parts[1:])
        snippet_num = "1"
    
    return model_name, version, snippet_num

def discover_yaml_files(directory: str) -> Dict[str, str]:
    """
    Discover all YAML files in the given directory.
    
    Returns:
        Dictionary mapping display names to file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    yaml_files = {}
    for filepath in dir_path.glob("*.yaml"):
        model_name, version, snippet = extract_model_info(filepath.name)
        display_name = f"{model_name}_{version}_snippet_{snippet}"
        yaml_files[display_name] = str(filepath)
    
    if not yaml_files:
        raise ValueError(f"No YAML files found in directory: {directory}")
    
    return yaml_files


def group_by_model_and_snippet(results_df: pd.DataFrame) -> Dict:
    """
    Group results by model and snippet for comparison.
    
    Returns:
        Dictionary with grouped data
    """
    grouped = {
        'by_model': {},
        'by_snippet': {},
        'by_model_version': {}
    }
    
    for _, row in results_df.iterrows():
        parts = row['Model'].split('_')
        model = parts[0]
        
        if 'snippet' in row['Model']:
            snippet = row['Model'].split('snippet_')[-1]
        else:
            snippet = '1'
        
        if model not in grouped['by_model']:
            grouped['by_model'][model] = []
        grouped['by_model'][model].append(row)
        
        if snippet not in grouped['by_snippet']:
            grouped['by_snippet'][snippet] = []
        grouped['by_snippet'][snippet].append(row)
    
    return grouped


def create_comparison_plots(df: pd.DataFrame, output_dir: str = '/home/claude'):
    """Create comprehensive comparison visualizations."""
    
    df['ModelName'] = df['Model'].apply(lambda x: x.split('_')[0])
    df['Snippet'] = df['Model'].apply(lambda x: x.split('snippet_')[-1] if 'snippet' in x else '1')
    
    model_colors = {
        'claude': '#FF6B6B',
        'gemini': '#4ECDC4', 
        'gpt': '#45B7D1',
        'grok': '#95E1D3',
        'deepseek': '#A569BD', 
        'llama': '#F39C12',
        'qwen': '#45B39D'
    }
    
    df['Color'] = df['ModelName'].map(model_colors)
    
    has_novelty = 'Novelty_Score' in df.columns
    
    if has_novelty:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('LLM YAML Output Analysis: Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    ax1 = fig.add_subplot(gs[0, 0])
    models = df['ModelName'].unique()
    snippets = sorted(df['Snippet'].unique())
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, snippet in enumerate(snippets):
        snippet_data = df[df['Snippet'] == snippet]
        quantities = [snippet_data[snippet_data['ModelName'] == m]['Quantity'].values[0] 
                     if len(snippet_data[snippet_data['ModelName'] == m]) > 0 else 0 
                     for m in models]
        ax1.bar(x + i*width, quantities, width, label=f'Snippet {snippet}', alpha=0.8)
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Total Items Extracted', fontweight='bold')
    ax1.set_title('Quantity Comparison by Model')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for i, snippet in enumerate(snippets):
        snippet_data = df[df['Snippet'] == snippet]
        avg_qualities = [snippet_data[snippet_data['ModelName'] == m]['Avg_Quality'].values[0] 
                        if len(snippet_data[snippet_data['ModelName'] == m]) > 0 else 0 
                        for m in models]
        ax2.bar(x + i*width, avg_qualities, width, label=f'Snippet {snippet}', alpha=0.8)
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Avg Fields per Item', fontweight='bold')
    ax2.set_title('Average Quality Comparison')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    for i, snippet in enumerate(snippets):
        snippet_data = df[df['Snippet'] == snippet]
        unique_keys = [snippet_data[snippet_data['ModelName'] == m]['Unique_Keys'].values[0] 
                      if len(snippet_data[snippet_data['ModelName'] == m]) > 0 else 0 
                      for m in models]
        ax3.bar(x + i*width, unique_keys, width, label=f'Snippet {snippet}', alpha=0.8)
    
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Number of Unique Field Types', fontweight='bold')
    ax3.set_title('Field Type Diversity')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    if has_novelty:
        ax_nov = fig.add_subplot(gs[0, 3])
        for i, snippet in enumerate(snippets):
            snippet_data = df[df['Snippet'] == snippet]
            novelty_scores = [snippet_data[snippet_data['ModelName'] == m]['Novelty_Score'].values[0] 
                             if len(snippet_data[snippet_data['ModelName'] == m]) > 0 else 0 
                             for m in models]
            ax_nov.bar(x + i*width, novelty_scores, width, label=f'Snippet {snippet}', alpha=0.8)
        
        ax_nov.set_xlabel('Model', fontweight='bold')
        ax_nov.set_ylabel('Novel Parameters', fontweight='bold')
        ax_nov.set_title('Novelty Score\n(Non-Standard Parameters)')
        ax_nov.set_xticks(x + width/2)
        ax_nov.set_xticklabels(models)
        ax_nov.legend()
        ax_nov.grid(axis='y', alpha=0.3)
    
    if has_novelty:
        ax4 = fig.add_subplot(gs[1, :2])
    else:
        ax4 = fig.add_subplot(gs[1, :2])
    
    for model in models:
        model_data = df[df['ModelName'] == model]
        ax4.scatter(model_data['Quantity'], model_data['Avg_Quality'], 
                   s=300, c=model_colors.get(model, '#999999'), 
                   alpha=0.6, edgecolors='black', linewidth=2,
                   label=model)
        
        for _, row in model_data.iterrows():
            ax4.annotate(f"S{row['Snippet']}", 
                        (row['Quantity'], row['Avg_Quality']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('Quantity (Total Items)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Quality (Avg Fields per Item)', fontweight='bold', fontsize=12)
    ax4.set_title('Quality vs Quantity Trade-off Analysis', fontsize=12, fontweight='bold')
    ax4.legend(title='Model', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    if has_novelty:
        ax5 = fig.add_subplot(gs[1, 2:])
    else:
        ax5 = fig.add_subplot(gs[1, 2])
    
    df['Norm_Quantity'] = (df['Quantity'] - df['Quantity'].min()) / (df['Quantity'].max() - df['Quantity'].min() + 0.001)
    df['Norm_Quality'] = (df['Avg_Quality'] - df['Avg_Quality'].min()) / (df['Avg_Quality'].max() - df['Avg_Quality'].min() + 0.001)
    
    if has_novelty:
        df['Norm_Novelty'] = (df['Novelty_Score'] - df['Novelty_Score'].min()) / (df['Novelty_Score'].max() - df['Novelty_Score'].min() + 0.001)
        df['Composite_Score'] = (df['Norm_Quantity'] + df['Norm_Quality'] + df['Norm_Novelty']) / 3
    else:
        df['Composite_Score'] = (df['Norm_Quantity'] + df['Norm_Quality']) / 2
    
    df_sorted = df.sort_values('Composite_Score', ascending=True)
    colors_list = [df_sorted.iloc[i]['Color'] for i in range(len(df_sorted))]
    
    ax5.barh(range(len(df_sorted)), df_sorted['Composite_Score'], color=colors_list, alpha=0.7)
    ax5.set_yticks(range(len(df_sorted)))
    ax5.set_yticklabels(df_sorted['Model'], fontsize=8)
    ax5.set_xlabel('Composite Score', fontweight='bold')
    title_suffix = ' + Novelty' if has_novelty else ''
    ax5.set_title(f'Overall Rankings\n(Quality + Quantity{title_suffix})', fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    if has_novelty:
        ax6 = fig.add_subplot(gs[2, :])
        heatmap_cols = ['Quantity', 'Avg_Quality', 'Max_Quality', 'Unique_Keys', 'Novelty_Score']
    else:
        ax6 = fig.add_subplot(gs[2, :])
        heatmap_cols = ['Quantity', 'Avg_Quality', 'Max_Quality', 'Unique_Keys']
    
    heatmap_data = df[['Model'] + heatmap_cols].set_index('Model')
    
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T, fmt='.2f', 
                cmap='YlOrRd', cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax6)
    ax6.set_title('Detailed Metrics Heatmap (annotated with actual values)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Model', fontweight='bold')
    ax6.set_ylabel('Metric', fontweight='bold')
    
    plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_dir}/comprehensive_analysis.png")
    
    if has_novelty:
        fig_nov, axes_nov = plt.subplots(2, 2, figsize=(16, 10))
        fig_nov.suptitle('Novelty Analysis: Novel vs Standard Parameters', fontsize=16, fontweight='bold')
        
        ax_n1 = axes_nov[0, 0]
        df_sorted_nov = df.sort_values('Novelty_Score', ascending=False)
        colors_n1 = [df_sorted_nov.iloc[i]['Color'] for i in range(len(df_sorted_nov))]
        ax_n1.bar(range(len(df_sorted_nov)), df_sorted_nov['Novelty_Score'], color=colors_n1, alpha=0.7)
        ax_n1.set_xticks(range(len(df_sorted_nov)))
        ax_n1.set_xticklabels(df_sorted_nov['Model'], rotation=45, ha='right', fontsize=8)
        ax_n1.set_ylabel('Novel Parameters Count', fontweight='bold')
        ax_n1.set_title('Novelty Score Ranking')
        ax_n1.grid(axis='y', alpha=0.3)
        
        ax_n2 = axes_nov[0, 1]
        df_sorted_ratio = df.sort_values('Novelty_Ratio', ascending=False)
        colors_n2 = [df_sorted_ratio.iloc[i]['Color'] for i in range(len(df_sorted_ratio))]
        ax_n2.bar(range(len(df_sorted_ratio)), df_sorted_ratio['Novelty_Ratio'], color=colors_n2, alpha=0.7)
        ax_n2.set_xticks(range(len(df_sorted_ratio)))
        ax_n2.set_xticklabels(df_sorted_ratio['Model'], rotation=45, ha='right', fontsize=8)
        ax_n2.set_ylabel('Novelty Ratio', fontweight='bold')
        ax_n2.set_title('Novelty Ratio (Novel/Total)')
        ax_n2.set_ylim([0, 1])
        ax_n2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(df_sorted_ratio['Novelty_Ratio']):
            ax_n2.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=8)
        
        ax_n3 = axes_nov[1, 0]
        x_pos = np.arange(len(df))
        ax_n3.bar(x_pos, df['Matched_Standard'], label='Standard', color='#3498db', alpha=0.8)
        ax_n3.bar(x_pos, df['Novelty_Score'], bottom=df['Matched_Standard'], 
                 label='Novel', color='#e74c3c', alpha=0.8)
        ax_n3.set_xticks(x_pos)
        ax_n3.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=8)
        ax_n3.set_ylabel('Parameter Count', fontweight='bold')
        ax_n3.set_title('Novel vs Standard Parameters Distribution')
        ax_n3.legend()
        ax_n3.grid(axis='y', alpha=0.3)
        
        ax_n4 = axes_nov[1, 1]
        for model in models:
            model_data = df[df['ModelName'] == model]
            ax_n4.scatter(model_data['Novelty_Score'], model_data['Avg_Quality'],
                         s=300, c=model_colors.get(model, '#999999'),
                         alpha=0.6, edgecolors='black', linewidth=2,
                         label=model)
            
            for _, row in model_data.iterrows():
                ax_n4.annotate(f"S{row['Snippet']}", 
                              (row['Novelty_Score'], row['Avg_Quality']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, fontweight='bold')
        
        ax_n4.set_xlabel('Novelty Score (Novel Parameters)', fontweight='bold')
        ax_n4.set_ylabel('Average Quality (Fields per Item)', fontweight='bold')
        ax_n4.set_title('Novelty vs Quality Relationship')
        ax_n4.legend(title='Model')
        ax_n4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/novelty_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Novelty analysis saved to: {output_dir}/novelty_analysis.png")
    
    fig2, axes2 = plt.subplots(1, len(snippets), figsize=(7*len(snippets), 6))
    if len(snippets) == 1:
        axes2 = [axes2]
    
    fig2.suptitle('Snippet-by-Snippet Comparison', fontsize=16, fontweight='bold')
    
    for idx, snippet in enumerate(snippets):
        ax = axes2[idx]
        snippet_data = df[df['Snippet'] == snippet].sort_values('Quantity', ascending=False)
        
        x_pos = np.arange(len(snippet_data))
        
        if has_novelty:
            width = 0.2
            ax.bar(x_pos - 1.5*width, snippet_data['Quantity'], width, label='Quantity', alpha=0.8, color='#3498db')
            ax.bar(x_pos - 0.5*width, snippet_data['Avg_Quality']*2, width, label='Avg Quality (×2)', alpha=0.8, color='#e74c3c')
            ax.bar(x_pos + 0.5*width, snippet_data['Unique_Keys'], width, label='Unique Keys', alpha=0.8, color='#2ecc71')
            ax.bar(x_pos + 1.5*width, snippet_data['Novelty_Score'], width, label='Novelty', alpha=0.8, color='#f39c12')
        else:
            width = 0.25
            ax.bar(x_pos - width, snippet_data['Quantity'], width, label='Quantity', alpha=0.8, color='#3498db')
            ax.bar(x_pos, snippet_data['Avg_Quality']*2, width, label='Avg Quality (×2)', alpha=0.8, color='#e74c3c')
            ax.bar(x_pos + width, snippet_data['Unique_Keys'], width, label='Unique Keys', alpha=0.8, color='#2ecc71')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(f'Snippet {snippet}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(snippet_data['ModelName'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/snippet_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Snippet comparison saved to: {output_dir}/snippet_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze YAML outputs from different LLM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_yaml_outputs.py /path/to/outputs_one_shot
  
  python analyze_yaml_outputs.py /path/to/outputs_one_shot -s /path/to/schema_dir
  
  python analyze_yaml_outputs.py ./outputs_one_shot -o ./results -s ./schemas
        """
    )
    parser.add_argument('directory', type=str, help='Directory containing YAML files to analyze')
    parser.add_argument('-o', '--output', type=str, default='/home/claude', 
                       help='Output directory for results (default: /home/claude)')
    parser.add_argument('-s', '--schema', type=str, default=None,
                       help='Directory containing JSON schema files for novelty scoring (optional)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LLM YAML OUTPUT ANALYZER")
    print("=" * 80)
    print(f"\nAnalyzing directory: {args.directory}")
    print(f"Output directory: {args.output}\n")
    
    schema_keys = None
    if args.schema:
        print(f"Schema directory: {args.schema}")
        try:
            schema_keys = extract_all_schema_keys(args.schema, verbose=True)
        except Exception as e:
            print(f"Warning: Could not extract schema keys: {e}")
            print("Continuing without novelty scoring...\n")
            schema_keys = None
    else:
        print("No schema directory provided - novelty scoring disabled")
        print("(Use -s/--schema to enable novelty scoring)\n")
    
    try:
        yaml_files = discover_yaml_files(args.directory)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} YAML files to analyze:\n")
    for name in sorted(yaml_files.keys()):
        print(f"  - {name}")
    print()
    
    results = []
    for name, filepath in sorted(yaml_files.items()):
        print(f"Analyzing: {name}...")
        try:
            quantity, avg_quality, max_quality, all_keys = analyze_yaml_file(filepath)
            
            result = {
                'Model': name,
                'Quantity': quantity,
                'Avg_Quality': avg_quality,
                'Max_Quality': max_quality,
                'Unique_Keys': len(all_keys),
                'Keys': sorted(all_keys)
            }
            
            if schema_keys is not None:
                novelty_metrics = compute_novelty_score(all_keys, schema_keys)
                result['Novelty_Score'] = novelty_metrics['novelty_score']
                result['Novelty_Ratio'] = novelty_metrics['novelty_ratio']
                result['Novel_Params'] = novelty_metrics['novel_params']
                result['Matched_Standard'] = novelty_metrics['matched_standard_params']
                
                print(f"  Novelty: {novelty_metrics['novelty_score']} novel params ({novelty_metrics['novelty_ratio']*100:.1f}%)")
            
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not results:
        print("No files were successfully analyzed!")
        sys.exit(1)
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print("\nDefinitions:")
    print("  Quantity:        Total number of items/parameters extracted")
    print("  Avg Quality:     Average number of unique fields per item")
    print("  Max Quality:     Maximum unique fields in any single item")
    print("  Unique Keys:     Total number of distinct field types used")
    if schema_keys is not None:
        print("  Novelty Score:   Number of parameters NOT in standard schema")
        print("  Novelty Ratio:   Proportion of novel to total parameters")
    print("\n" + "=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Quantity:        {row['Quantity']}")
        print(f"  Avg Quality:     {row['Avg_Quality']:.2f}")
        print(f"  Max Quality:     {row['Max_Quality']}")
        print(f"  Unique Keys:     {row['Unique_Keys']}")
        if 'Novelty_Score' in row:
            print(f"  Novelty Score:   {row['Novelty_Score']}")
            print(f"  Novelty Ratio:   {row['Novelty_Ratio']:.2%}")
            print(f"  Standard Match:  {row['Matched_Standard']}")
        print(f"  Field types:     {', '.join(row['Keys'][:8])}")
        if len(row['Keys']) > 8:
            print(f"                   ... and {len(row['Keys']) - 8} more")
    
    print("\n" + "=" * 80)
    
    grouped = group_by_model_and_snippet(df)
    
    print("\nGROUPED ANALYSIS:")
    print("-" * 80)
    print("\nBy Model:")
    for model, rows in grouped['by_model'].items():
        print(f"\n  {model.upper()}:")
        for row in rows:
            nov_str = f", Nov={row.get('Novelty_Score', 'N/A')}" if 'Novelty_Score' in row else ""
            print(f"    {row['Model']}: Qty={row['Quantity']}, AvgQ={row['Avg_Quality']:.2f}{nov_str}")
    
    print("\n" + "=" * 80)
    
    print("\nGenerating visualizations...")
    create_comparison_plots(df, args.output)
    
    has_novelty = 'Novelty_Score' in df.columns
    
    if has_novelty:
        table_height = max(6, len(df) * 0.5)
        fig_table, ax_table = plt.subplots(figsize=(16, table_height))
        col_labels = ['Model', 'Quantity\n(Items)', 'Avg Quality\n(Fields/Item)', 
                     'Max Quality\n(Fields)', 'Unique\nField Types', 
                     'Novelty\nScore', 'Novelty\nRatio']
        col_widths = [0.25, 0.1, 0.12, 0.1, 0.12, 0.1, 0.1]
    else:
        table_height = max(6, len(df) * 0.5)
        fig_table, ax_table = plt.subplots(figsize=(14, table_height))
        col_labels = ['Model', 'Quantity\n(Items)', 'Avg Quality\n(Fields/Item)', 
                     'Max Quality\n(Fields)', 'Unique\nField Types']
        col_widths = [0.35, 0.15, 0.15, 0.15, 0.15]
    
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = []
    for _, row in df.iterrows():
        if has_novelty:
            table_data.append([
                row['Model'],
                str(row['Quantity']),
                f"{row['Avg_Quality']:.2f}",
                str(row['Max_Quality']),
                str(row['Unique_Keys']),
                str(row['Novelty_Score']),
                f"{row['Novelty_Ratio']:.2%}"
            ])
        else:
            table_data.append([
                row['Model'],
                str(row['Quantity']),
                f"{row['Avg_Quality']:.2f}",
                str(row['Max_Quality']),
                str(row['Unique_Keys'])
            ])
    
    table = ax_table.table(cellText=table_data,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center',
                          colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Summary Comparison Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{args.output}/summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Summary table saved to: {args.output}/summary_table.png")
    
    if has_novelty:
        df_export = df.drop(['Keys', 'Novel_Params'], axis=1)
    else:
        df_export = df.drop('Keys', axis=1)
    
    df_export.to_csv(f'{args.output}/analysis_results.csv', index=False)
    print(f"Detailed results saved to: {args.output}/analysis_results.csv")
    
    with open(f'{args.output}/detailed_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Model']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Quantity:     {row['Quantity']}\n")
            f.write(f"Avg Quality:  {row['Avg_Quality']:.2f}\n")
            f.write(f"Max Quality:  {row['Max_Quality']}\n")
            f.write(f"Unique Keys:  {row['Unique_Keys']}\n")
            
            if 'Novelty_Score' in row:
                f.write(f"Novelty Score: {row['Novelty_Score']}\n")
                f.write(f"Novelty Ratio: {row['Novelty_Ratio']:.2%}\n")
                f.write(f"Standard Match: {row['Matched_Standard']}\n")
                f.write(f"\nNovel Parameters:\n")
                for param in row['Novel_Params']:
                    f.write(f"  * {param}\n")
            
            f.write(f"\nAll Field Types:\n")
            for key in row['Keys']:
                f.write(f"  - {key}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Detailed report saved to: {args.output}/detailed_report.txt")
    
    if schema_keys is not None:
        with open(f'{args.output}/standard_schema_keys.txt', 'w') as f:
            f.write("# Standard Schema Parameters\n")
            f.write(f"# Total: {len(schema_keys)}\n\n")
            for key in sorted(schema_keys):
                f.write(f"{key}\n")
        print(f"Schema keys reference saved to: {args.output}/standard_schema_keys.txt")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
