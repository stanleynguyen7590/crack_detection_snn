#!/usr/bin/env python3
"""
SDNET2018 Dataset Analysis Script
Analyzes class distribution for multi-class crack detection implementation

This script provides comprehensive analysis of the SDNET2018 dataset structure
to inform multi-class implementation decisions.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import json

def count_files_in_directory(directory):
    """Count number of files in a directory"""
    try:
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    except OSError:
        return 0

def analyze_sdnet2018_distribution(data_dir):
    """
    Analyze SDNET2018 dataset distribution across all categories
    
    Expected structure:
    data_dir/
    ├── D/ (Decks)
    │   ├── CD/ (Cracked)
    │   └── UD/ (Uncracked)
    ├── P/ (Pavements)
    │   ├── CP/ (Cracked)
    │   └── UP/ (Uncracked)
    └── W/ (Walls)
        ├── CW/ (Cracked)
        └── UW/ (Uncracked)
    """
    
    print("🔍 Analyzing SDNET2018 Dataset Distribution")
    print("=" * 60)
    
    # Define categories
    categories = {
        'CD': 'Cracked Decks',
        'UD': 'Uncracked Decks', 
        'CP': 'Cracked Pavements',
        'UP': 'Uncracked Pavements',
        'CW': 'Cracked Walls',
        'UW': 'Uncracked Walls'
    }
    
    # Count samples in each category
    category_counts = {}
    structure_counts = defaultdict(int)
    crack_counts = defaultdict(int)
    
    for structure in ['D', 'P', 'W']:
        structure_dir = os.path.join(data_dir, structure)
        if not os.path.exists(structure_dir):
            print(f"⚠️  Warning: Structure directory {structure} not found")
            continue
            
        print(f"\n📁 Structure: {structure}")
        print("-" * 30)
        
        for crack_status in ['C', 'U']:
            category = f"{crack_status}{structure}"
            category_dir = os.path.join(structure_dir, category)
            
            if os.path.exists(category_dir):
                count = count_files_in_directory(category_dir)
                category_counts[category] = count
                structure_counts[structure] += count
                crack_counts[crack_status] += count
                
                print(f"  {category} ({categories[category]}): {count:,} images")
            else:
                print(f"  {category} ({categories[category]}): Directory not found")
                category_counts[category] = 0
    
    return category_counts, structure_counts, crack_counts

def calculate_class_weights(category_counts):
    """Calculate class weights for balanced training"""
    total_samples = sum(category_counts.values())
    num_classes = len(category_counts)
    
    # Calculate inverse frequency weights
    class_weights = {}
    for category, count in category_counts.items():
        if count > 0:
            weight = total_samples / (num_classes * count)
            class_weights[category] = weight
        else:
            class_weights[category] = 0
    
    return class_weights

def analyze_class_schemes(category_counts, structure_counts, crack_counts):
    """Analyze different class scheme options"""
    
    print("\n🎯 Class Scheme Analysis")
    print("=" * 60)
    
    # 1. Current Binary Classification
    print("\n1. Current Binary Classification (Cracked vs Uncracked)")
    print("-" * 50)
    total_cracked = crack_counts['C']
    total_uncracked = crack_counts['U']
    total_samples = total_cracked + total_uncracked
    
    print(f"  Cracked: {total_cracked:,} ({total_cracked/total_samples*100:.1f}%)")
    print(f"  Uncracked: {total_uncracked:,} ({total_uncracked/total_samples*100:.1f}%)")
    
    binary_imbalance = max(total_cracked, total_uncracked) / min(total_cracked, total_uncracked)
    print(f"  Imbalance ratio: {binary_imbalance:.2f}:1")
    
    # 2. 3-Class Structure-Based
    print("\n2. 3-Class Structure-Based (Deck/Pavement/Wall Cracks)")
    print("-" * 50)
    structure_cracks = {
        'Deck Cracks': category_counts.get('CD', 0),
        'Pavement Cracks': category_counts.get('CP', 0),
        'Wall Cracks': category_counts.get('CW', 0)
    }
    
    total_cracks = sum(structure_cracks.values())
    for structure, count in structure_cracks.items():
        percentage = (count / total_cracks * 100) if total_cracks > 0 else 0
        print(f"  {structure}: {count:,} ({percentage:.1f}%)")
    
    if total_cracks > 0:
        max_crack = max(structure_cracks.values())
        min_crack = min(v for v in structure_cracks.values() if v > 0)
        crack_imbalance = max_crack / min_crack if min_crack > 0 else float('inf')
        print(f"  Imbalance ratio: {crack_imbalance:.2f}:1")
    
    # 3. 6-Class Full Classification
    print("\n3. 6-Class Full Classification (All Categories)")
    print("-" * 50)
    
    categories_full = {
        'CD': 'Cracked Decks',
        'UD': 'Uncracked Decks',
        'CP': 'Cracked Pavements', 
        'UP': 'Uncracked Pavements',
        'CW': 'Cracked Walls',
        'UW': 'Uncracked Walls'
    }
    
    for category, description in categories_full.items():
        count = category_counts.get(category, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {category} ({description}): {count:,} ({percentage:.1f}%)")
    
    if total_samples > 0:
        max_class = max(category_counts.values())
        min_class = min(v for v in category_counts.values() if v > 0)
        full_imbalance = max_class / min_class if min_class > 0 else float('inf')
        print(f"  Imbalance ratio: {full_imbalance:.2f}:1")
    
    return {
        'binary_imbalance': binary_imbalance,
        'structure_cracks': structure_cracks,
        'full_categories': category_counts
    }

def create_visualizations(category_counts, structure_counts, crack_counts, output_dir="results"):
    """Create visualizations of dataset distribution"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SDNET2018 Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Binary Classification (Cracked vs Uncracked)
    ax1 = axes[0, 0]
    binary_data = [crack_counts['C'], crack_counts['U']]
    binary_labels = ['Cracked', 'Uncracked']
    colors1 = ['#ff6b6b', '#4ecdc4']
    
    wedges, texts, autotexts = ax1.pie(binary_data, labels=binary_labels, autopct='%1.1f%%', 
                                      colors=colors1, startangle=90)
    ax1.set_title('Current Binary Classification\n(Cracked vs Uncracked)', fontweight='bold')
    
    # 2. Structure Distribution
    ax2 = axes[0, 1]
    structure_data = [structure_counts['D'], structure_counts['P'], structure_counts['W']]
    structure_labels = ['Decks', 'Pavements', 'Walls']
    colors2 = ['#95e1d3', '#fce38a', '#f38ba8']
    
    bars = ax2.bar(structure_labels, structure_data, color=colors2)
    ax2.set_title('Distribution by Structure Type', fontweight='bold')
    ax2.set_ylabel('Number of Images')
    
    # Add value labels on bars
    for bar, value in zip(bars, structure_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(structure_data)*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 6-Class Full Distribution
    ax3 = axes[1, 0]
    categories_order = ['CD', 'UD', 'CP', 'UP', 'CW', 'UW']
    full_data = [category_counts.get(cat, 0) for cat in categories_order]
    colors3 = ['#ff9999', '#ffcc99', '#99ff99', '#99ffcc', '#99ccff', '#cc99ff']
    
    bars = ax3.bar(categories_order, full_data, color=colors3)
    ax3.set_title('6-Class Full Distribution', fontweight='bold')
    ax3.set_ylabel('Number of Images')
    ax3.set_xlabel('Categories')
    
    # Add value labels on bars
    for bar, value in zip(bars, full_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(full_data)*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', rotation=45)
    
    # 4. Class Imbalance Comparison
    ax4 = axes[1, 1]
    
    # Calculate imbalance ratios
    binary_imbalance = max(crack_counts['C'], crack_counts['U']) / min(crack_counts['C'], crack_counts['U'])
    
    structure_cracks = [category_counts.get('CD', 0), category_counts.get('CP', 0), category_counts.get('CW', 0)]
    structure_imbalance = max(structure_cracks) / min(s for s in structure_cracks if s > 0) if any(s > 0 for s in structure_cracks) else 1
    
    full_imbalance = max(category_counts.values()) / min(v for v in category_counts.values() if v > 0) if any(v > 0 for v in category_counts.values()) else 1
    
    schemes = ['Binary\n(2-class)', '3-Class\n(Structures)', '6-Class\n(Full)']
    imbalances = [binary_imbalance, structure_imbalance, full_imbalance]
    colors4 = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax4.bar(schemes, imbalances, color=colors4)
    ax4.set_title('Class Imbalance Comparison', fontweight='bold')
    ax4.set_ylabel('Imbalance Ratio (Max:Min)')
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
    
    # Add value labels
    for bar, value in zip(bars, imbalances):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.1f}:1', ha='center', va='bottom', fontweight='bold')
    
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'dataset_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    return output_path

def generate_recommendations(analysis_results, category_counts):
    """Generate recommendations based on analysis"""
    
    print("\n💡 Recommendations")
    print("=" * 60)
    
    binary_imbalance = analysis_results['binary_imbalance']
    structure_cracks = analysis_results['structure_cracks']
    
    # Determine structure imbalance
    max_structure = max(structure_cracks.values())
    min_structure = min(v for v in structure_cracks.values() if v > 0)
    structure_imbalance = max_structure / min_structure if min_structure > 0 else float('inf')
    
    # Determine full imbalance
    max_full = max(category_counts.values())
    min_full = min(v for v in category_counts.values() if v > 0)
    full_imbalance = max_full / min_full if min_full > 0 else float('inf')
    
    print(f"1. Class Balance Assessment:")
    print(f"   - Binary classification imbalance: {binary_imbalance:.2f}:1 {'✅ Acceptable' if binary_imbalance < 3 else '⚠️ Moderate' if binary_imbalance < 5 else '❌ Severe'}")
    print(f"   - 3-class structure imbalance: {structure_imbalance:.2f}:1 {'✅ Acceptable' if structure_imbalance < 3 else '⚠️ Moderate' if structure_imbalance < 5 else '❌ Severe'}")
    print(f"   - 6-class full imbalance: {full_imbalance:.2f}:1 {'✅ Acceptable' if full_imbalance < 3 else '⚠️ Moderate' if full_imbalance < 5 else '❌ Severe'}")
    
    print(f"\n2. Recommended Implementation Order:")
    if binary_imbalance < 3:
        print("   ✅ Start with current binary classification (well-balanced)")
    elif binary_imbalance < 5:
        print("   ⚠️ Current binary classification (moderate imbalance - consider class weights)")
    else:
        print("   ❌ Current binary classification (severe imbalance - needs balancing)")
    
    if structure_imbalance < 5:
        print("   ✅ 3-class structure-based classification (feasible)")
    else:
        print("   ⚠️ 3-class structure-based classification (needs careful balancing)")
    
    if full_imbalance < 5:
        print("   ✅ 6-class full classification (feasible with balancing)")
    else:
        print("   ❌ 6-class full classification (challenging - needs advanced balancing)")
    
    print(f"\n3. Class Weighting Strategy:")
    if max(binary_imbalance, structure_imbalance, full_imbalance) > 2:
        print("   📋 Recommended techniques:")
        print("   - Use class weights in loss function")
        print("   - Implement WeightedRandomSampler")
        print("   - Consider oversampling minority classes")
        print("   - Use stratified cross-validation")

def save_analysis_results(category_counts, structure_counts, crack_counts, class_weights, output_dir="results"):
    """Save analysis results to JSON file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'dataset_analysis': {
            'category_counts': category_counts,
            'structure_counts': dict(structure_counts),
            'crack_counts': dict(crack_counts),
            'total_samples': sum(category_counts.values()),
            'analysis_timestamp': str(np.datetime64('now'))
        },
        'class_weights': {
            'binary': {
                'cracked': crack_counts['U'] / crack_counts['C'] if crack_counts['C'] > 0 else 1,
                'uncracked': crack_counts['C'] / crack_counts['U'] if crack_counts['U'] > 0 else 1
            },
            'full_6_class': class_weights
        },
        'imbalance_ratios': {
            'binary': max(crack_counts['C'], crack_counts['U']) / min(crack_counts['C'], crack_counts['U']),
            'structure_based': max(category_counts.get('CD', 0), category_counts.get('CP', 0), category_counts.get('CW', 0)) / min(v for v in [category_counts.get('CD', 0), category_counts.get('CP', 0), category_counts.get('CW', 0)] if v > 0),
            'full_6_class': max(category_counts.values()) / min(v for v in category_counts.values() if v > 0)
        }
    }
    
    output_path = os.path.join(output_dir, 'dataset_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Analysis results saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze SDNET2018 dataset distribution for multi-class implementation')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to SDNET2018 dataset directory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save analysis results (default: results)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip creating visualizations')
    
    args = parser.parse_args()
    
    # Verify dataset directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Dataset directory '{args.data_dir}' not found")
        return
    
    # Analyze dataset distribution
    category_counts, structure_counts, crack_counts = analyze_sdnet2018_distribution(args.data_dir)
    
    # Calculate class weights
    class_weights = calculate_class_weights(category_counts)
    
    # Analyze class schemes
    analysis_results = analyze_class_schemes(category_counts, structure_counts, crack_counts)
    
    # Create visualizations
    if not args.no_visualization:
        create_visualizations(category_counts, structure_counts, crack_counts, args.output_dir)
    
    # Generate recommendations
    generate_recommendations(analysis_results, category_counts)
    
    # Save results
    save_analysis_results(category_counts, structure_counts, crack_counts, class_weights, args.output_dir)
    
    print(f"\n🎉 Analysis complete! Results saved to '{args.output_dir}' directory")

if __name__ == "__main__":
    main()