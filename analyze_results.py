#!/usr/bin/env python3
"""
Results Analysis Script for Crack Detection Experiments
Analyzes and compares results from different experimental runs
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Optional

def find_all_result_files(base_dir: str = 'results') -> Dict[str, List[str]]:
    """Find all result JSON files organized by experiment type"""
    
    results = {
        'snn_experiments': [],
        'cnn_experiments': [],
        'threshold_tuning': [],
        'summaries': []
    }
    
    if not os.path.exists(base_dir):
        print(f"Results directory {base_dir} not found")
        return results
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Categorize files
                if 'spiking' in file.lower() or 'snn' in file.lower():
                    results['snn_experiments'].append(file_path)
                elif any(arch in file.lower() for arch in ['resnet', 'inception', 'xception', 'baseline']):
                    results['cnn_experiments'].append(file_path)
                elif 'threshold_tuning' in file.lower():
                    results['threshold_tuning'].append(file_path)
                elif 'summary' in file.lower():
                    results['summaries'].append(file_path)
    
    return results

def load_experiment_data(file_paths: List[str]) -> List[Dict]:
    """Load experiment data from JSON files"""
    
    experiments = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Add metadata
            data['file_path'] = file_path
            data['file_name'] = os.path.basename(file_path)
            data['experiment_dir'] = os.path.basename(os.path.dirname(file_path))
            
            experiments.append(data)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return experiments

def analyze_performance_metrics(experiments: List[Dict], experiment_type: str = "Experiments") -> pd.DataFrame:
    """Analyze performance metrics across experiments"""
    
    print(f"\n📊 ANALYZING {experiment_type.upper()}")
    print("=" * 60)
    
    if not experiments:
        print("No experiments found for analysis")
        return pd.DataFrame()
    
    # Extract key metrics
    analysis_data = []
    
    for exp in experiments:
        # Determine experiment details from file path and content
        exp_info = {
            'experiment': exp.get('file_name', 'unknown'),
            'directory': exp.get('experiment_dir', 'unknown'),
            'model_type': 'SNN' if any(keyword in exp.get('file_name', '').lower() 
                                     for keyword in ['snn', 'spiking']) else 'CNN'
        }
        
        # Extract performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
        
        for metric in metrics:
            value = exp.get(metric, None)
            if value is not None:
                exp_info[metric] = float(value)
        
        # Extract training info if available
        if 'training_time' in exp:
            exp_info['training_time'] = float(exp['training_time'])
        
        if 'best_val_acc' in exp:
            exp_info['best_val_acc'] = float(exp['best_val_acc'])
        
        # Extract architecture info
        if 'architecture' in exp:
            exp_info['architecture'] = exp['architecture']
        
        analysis_data.append(exp_info)
    
    # Create DataFrame
    df = pd.DataFrame(analysis_data)
    
    if df.empty:
        print("No valid data found for analysis")
        return df
    
    print(f"Found {len(df)} experiments")
    print(f"Columns available: {list(df.columns)}")
    
    # Display summary statistics
    if 'f1_score' in df.columns:
        print(f"\nF1-Score Statistics:")
        print(f"  Mean: {df['f1_score'].mean():.4f}")
        print(f"  Std:  {df['f1_score'].std():.4f}")
        print(f"  Min:  {df['f1_score'].min():.4f}")
        print(f"  Max:  {df['f1_score'].max():.4f}")
    
    if 'accuracy' in df.columns:
        print(f"\nAccuracy Statistics:")
        print(f"  Mean: {df['accuracy'].mean():.4f}")
        print(f"  Std:  {df['accuracy'].std():.4f}")
        print(f"  Min:  {df['accuracy'].min():.4f}")
        print(f"  Max:  {df['accuracy'].max():.4f}")
    
    return df

def analyze_threshold_tuning_results(threshold_files: List[str]):
    """Analyze threshold tuning effectiveness"""
    
    print(f"\n🎯 ANALYZING THRESHOLD TUNING RESULTS")
    print("=" * 60)
    
    if not threshold_files:
        print("No threshold tuning results found")
        return
    
    improvements = []
    
    for file_path in threshold_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract improvement metrics
            default_metrics = data.get('default_threshold_metrics', {})
            optimal_metrics = data.get('optimal_metrics', {})
            
            if default_metrics and optimal_metrics:
                improvement = {
                    'file': os.path.basename(file_path),
                    'optimal_threshold': data.get('optimal_threshold', 0.5),
                    'optimization_metric': data.get('optimization_metric', 'unknown')
                }
                
                # Calculate improvements
                metrics = ['f1', 'precision', 'recall', 'specificity', 'balanced_accuracy']
                for metric in metrics:
                    default_val = default_metrics.get(metric, 0)
                    optimal_val = optimal_metrics.get(metric, 0)
                    improvement[f'{metric}_improvement'] = optimal_val - default_val
                    improvement[f'{metric}_default'] = default_val
                    improvement[f'{metric}_optimal'] = optimal_val
                
                improvements.append(improvement)
                
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
    
    if improvements:
        # Create summary
        df_thresh = pd.DataFrame(improvements)
        
        print(f"Found {len(improvements)} threshold tuning experiments")
        print(f"\nThreshold Statistics:")
        print(f"  Mean optimal threshold: {df_thresh['optimal_threshold'].mean():.3f}")
        print(f"  Threshold range: {df_thresh['optimal_threshold'].min():.3f} - {df_thresh['optimal_threshold'].max():.3f}")
        
        print(f"\nF1-Score Improvements:")
        f1_improvements = df_thresh['f1_improvement']
        print(f"  Mean improvement: {f1_improvements.mean():.4f}")
        print(f"  Best improvement: {f1_improvements.max():.4f}")
        print(f"  Experiments with >0.01 improvement: {len(f1_improvements[f1_improvements > 0.01])}")
        
        return df_thresh
    
    return None

def create_comparison_plots(snn_df: pd.DataFrame, cnn_df: pd.DataFrame, output_dir: str = '.'):
    """Create comparison plots between SNN and CNN results"""
    
    print(f"\n📊 CREATING COMPARISON PLOTS")
    print("=" * 60)
    
    if snn_df.empty and cnn_df.empty:
        print("No data available for plotting")
        return
    
    # Combine dataframes
    all_data = []
    
    if not snn_df.empty:
        snn_data = snn_df.copy()
        snn_data['model_type'] = 'SNN'
        all_data.append(snn_data)
    
    if not cnn_df.empty:
        cnn_data = cnn_df.copy()
        cnn_data['model_type'] = 'CNN'
        all_data.append(cnn_data)
    
    if not all_data:
        print("No valid data for plotting")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNN vs CNN Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: F1-Score comparison
    if 'f1_score' in combined_df.columns:
        sns.boxplot(data=combined_df, x='model_type', y='f1_score', ax=axes[0, 0])
        axes[0, 0].set_title('F1-Score Comparison')
        axes[0, 0].set_ylabel('F1-Score')
    
    # Plot 2: Accuracy comparison
    if 'accuracy' in combined_df.columns:
        sns.boxplot(data=combined_df, x='model_type', y='accuracy', ax=axes[0, 1])
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy')
    
    # Plot 3: Precision vs Recall
    if 'precision' in combined_df.columns and 'recall' in combined_df.columns:
        for model_type in combined_df['model_type'].unique():
            model_data = combined_df[combined_df['model_type'] == model_type]
            axes[1, 0].scatter(model_data['recall'], model_data['precision'], 
                              label=model_type, alpha=0.7, s=50)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training time comparison (if available)
    if 'training_time' in combined_df.columns:
        sns.boxplot(data=combined_df, x='model_type', y='training_time', ax=axes[1, 1])
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].set_ylabel('Training Time (seconds)')
    else:
        # Alternative: show architecture comparison for CNNs
        if 'architecture' in combined_df.columns:
            cnn_arch_data = combined_df[combined_df['model_type'] == 'CNN']
            if not cnn_arch_data.empty and 'f1_score' in cnn_arch_data.columns:
                sns.boxplot(data=cnn_arch_data, x='architecture', y='f1_score', ax=axes[1, 1])
                axes[1, 1].set_title('CNN Architecture Comparison')
                axes[1, 1].set_ylabel('F1-Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {output_path}")
    
    try:
        plt.show()
    except:
        print("Cannot display plot (no GUI available)")
    
    plt.close()

def generate_summary_report(snn_df: pd.DataFrame, cnn_df: pd.DataFrame, 
                           threshold_df: Optional[pd.DataFrame] = None, 
                           output_file: str = 'analysis_report.txt'):
    """Generate a comprehensive summary report"""
    
    print(f"\n📝 GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    with open(output_file, 'w') as f:
        f.write("CRACK DETECTION EXPERIMENTS - ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # SNN Results Summary
        f.write("SNN EXPERIMENTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        if not snn_df.empty:
            f.write(f"Number of SNN experiments: {len(snn_df)}\n")
            if 'f1_score' in snn_df.columns:
                f.write(f"Average F1-Score: {snn_df['f1_score'].mean():.4f} ± {snn_df['f1_score'].std():.4f}\n")
                f.write(f"Best F1-Score: {snn_df['f1_score'].max():.4f}\n")
            if 'accuracy' in snn_df.columns:
                f.write(f"Average Accuracy: {snn_df['accuracy'].mean():.4f} ± {snn_df['accuracy'].std():.4f}\n")
                f.write(f"Best Accuracy: {snn_df['accuracy'].max():.4f}\n")
        else:
            f.write("No SNN experiments found\n")
        f.write("\n")
        
        # CNN Results Summary
        f.write("CNN EXPERIMENTS SUMMARY\n")
        f.write("-" * 40 + "\n")
        if not cnn_df.empty:
            f.write(f"Number of CNN experiments: {len(cnn_df)}\n")
            if 'f1_score' in cnn_df.columns:
                f.write(f"Average F1-Score: {cnn_df['f1_score'].mean():.4f} ± {cnn_df['f1_score'].std():.4f}\n")
                f.write(f"Best F1-Score: {cnn_df['f1_score'].max():.4f}\n")
            if 'accuracy' in cnn_df.columns:
                f.write(f"Average Accuracy: {cnn_df['accuracy'].mean():.4f} ± {cnn_df['accuracy'].std():.4f}\n")
                f.write(f"Best Accuracy: {cnn_df['accuracy'].max():.4f}\n")
            if 'architecture' in cnn_df.columns:
                f.write(f"Architectures tested: {list(cnn_df['architecture'].unique())}\n")
        else:
            f.write("No CNN experiments found\n")
        f.write("\n")
        
        # Threshold Tuning Summary
        if threshold_df is not None and not threshold_df.empty:
            f.write("THRESHOLD TUNING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of threshold tuning experiments: {len(threshold_df)}\n")
            f.write(f"Average optimal threshold: {threshold_df['optimal_threshold'].mean():.3f}\n")
            f.write(f"Average F1 improvement: {threshold_df['f1_improvement'].mean():.4f}\n")
            f.write(f"Best F1 improvement: {threshold_df['f1_improvement'].max():.4f}\n")
            f.write(f"Experiments with significant improvement (>0.01): {len(threshold_df[threshold_df['f1_improvement'] > 0.01])}\n")
        else:
            f.write("THRESHOLD TUNING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write("No threshold tuning results found\n")
        f.write("\n")
        
        # Overall Conclusions
        f.write("CONCLUSIONS\n")
        f.write("-" * 40 + "\n")
        
        if not snn_df.empty and not cnn_df.empty:
            snn_best_f1 = snn_df['f1_score'].max() if 'f1_score' in snn_df.columns else 0
            cnn_best_f1 = cnn_df['f1_score'].max() if 'f1_score' in cnn_df.columns else 0
            
            if snn_best_f1 > cnn_best_f1:
                f.write(f"✅ SNN achieved better peak performance (F1: {snn_best_f1:.4f} vs {cnn_best_f1:.4f})\n")
            elif cnn_best_f1 > snn_best_f1:
                f.write(f"✅ CNN achieved better peak performance (F1: {cnn_best_f1:.4f} vs {snn_best_f1:.4f})\n")
            else:
                f.write("➡️ SNN and CNN achieved similar peak performance\n")
        
        if threshold_df is not None and not threshold_df.empty:
            significant_improvements = len(threshold_df[threshold_df['f1_improvement'] > 0.01])
            if significant_improvements > 0:
                f.write(f"✅ Threshold tuning provided significant improvements in {significant_improvements} experiments\n")
            else:
                f.write("➡️ Threshold tuning provided minimal improvements (dataset may be well-balanced)\n")
    
    print(f"📝 Summary report saved to: {output_file}")

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze crack detection experiment results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    print("🔍 CRACK DETECTION RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Analyzing results from: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Find all result files
    result_files = find_all_result_files(args.results_dir)
    
    print(f"\nFound result files:")
    for category, files in result_files.items():
        print(f"  {category}: {len(files)} files")
    
    # Load and analyze experiments
    snn_experiments = load_experiment_data(result_files['snn_experiments'])
    cnn_experiments = load_experiment_data(result_files['cnn_experiments'])
    
    # Analyze performance
    snn_df = analyze_performance_metrics(snn_experiments, "SNN Experiments")
    cnn_df = analyze_performance_metrics(cnn_experiments, "CNN Experiments")
    
    # Analyze threshold tuning
    threshold_df = analyze_threshold_tuning_results(result_files['threshold_tuning'])
    
    # Create comparison plots
    if not snn_df.empty or not cnn_df.empty:
        create_comparison_plots(snn_df, cnn_df, args.output_dir)
    
    # Generate summary report
    report_path = os.path.join(args.output_dir, 'analysis_report.txt')
    generate_summary_report(snn_df, cnn_df, threshold_df, report_path)
    
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"Check {args.output_dir} for:")
    print("  - performance_comparison.png (comparison plots)")
    print("  - analysis_report.txt (detailed summary)")

if __name__ == "__main__":
    main()