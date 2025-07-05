#!/usr/bin/env python3
"""
Threshold Tuning Module for Crack Detection
Optimizes classification threshold for imbalanced binary classification

This module finds the optimal threshold for binary classification that maximizes
a chosen metric (F1-score, balanced accuracy, etc.) on validation data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef
)
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class ThresholdTuner:
    """
    Optimize classification threshold for binary classification
    """
    
    def __init__(self, 
                 optimization_metric: str = 'f1',
                 threshold_range: Tuple[float, float] = (0.1, 0.9),
                 threshold_steps: int = 17,
                 class_names: List[str] = None):
        """
        Initialize threshold tuner
        
        Args:
            optimization_metric: Metric to optimize ('f1', 'balanced_accuracy', 'mcc', 'precision', 'recall')
            threshold_range: Range of thresholds to test
            threshold_steps: Number of threshold values to test
            class_names: Names for the two classes [negative, positive]
        """
        
        self.optimization_metric = optimization_metric
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.class_names = class_names or ['Uncracked', 'Cracked']
        
        # Generate threshold values to test
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_steps)
        
        # Results storage
        self.results = {}
        self.optimal_threshold = 0.5
        self.optimal_metrics = {}
        
        # Available optimization metrics
        self.metric_functions = {
            'f1': f1_score,
            'balanced_accuracy': balanced_accuracy_score,
            'mcc': matthews_corrcoef,
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score
        }
        
        if optimization_metric not in self.metric_functions:
            raise ValueError(f"Optimization metric must be one of {list(self.metric_functions.keys())}")
    
    def tune_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Find optimal threshold using validation data
        
        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary with optimal threshold and performance metrics
        """
        
        print(f"Tuning threshold to optimize {self.optimization_metric}...")
        print(f"Testing {len(self.thresholds)} threshold values from {self.threshold_range[0]} to {self.threshold_range[1]}")
        
        # Store metrics for each threshold
        threshold_metrics = {
            'thresholds': self.thresholds.tolist(),
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'accuracies': [],
            'balanced_accuracies': [],
            'mcc_scores': [],
            'specificities': [],
            'optimization_values': []
        }
        
        # Test each threshold
        for threshold in self.thresholds:
            # Make predictions with current threshold
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # Store metrics
            threshold_metrics['f1_scores'].append(metrics['f1'])
            threshold_metrics['precisions'].append(metrics['precision'])
            threshold_metrics['recalls'].append(metrics['recall'])
            threshold_metrics['accuracies'].append(metrics['accuracy'])
            threshold_metrics['balanced_accuracies'].append(metrics['balanced_accuracy'])
            threshold_metrics['mcc_scores'].append(metrics['mcc'])
            threshold_metrics['specificities'].append(metrics['specificity'])
            
            # Store optimization metric value
            optimization_value = metrics[self.optimization_metric]
            threshold_metrics['optimization_values'].append(optimization_value)
        
        # Find optimal threshold
        optimal_idx = np.argmax(threshold_metrics['optimization_values'])
        self.optimal_threshold = self.thresholds[optimal_idx]
        
        # Calculate optimal metrics
        y_pred_optimal = (y_prob >= self.optimal_threshold).astype(int)
        self.optimal_metrics = self._calculate_metrics(y_true, y_pred_optimal)
        
        # Store results
        self.results = {
            'optimal_threshold': float(self.optimal_threshold),
            'optimization_metric': self.optimization_metric,
            'optimal_metrics': self.optimal_metrics,
            'threshold_sweep': threshold_metrics,
            'default_threshold_metrics': self._calculate_metrics(y_true, (y_prob >= 0.5).astype(int)),
            'class_names': self.class_names,
            'dataset_info': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(len(y_true) - np.sum(y_true)),
                'class_imbalance_ratio': float(np.sum(y_true == 0) / np.sum(y_true)) if np.sum(y_true) > 0 else float('inf')
            }
        }
        
        self._print_results()
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics for binary classification"""
        
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            # All predictions are the same class
            if y_pred[0] == 1:
                # All predicted positive
                precision = np.mean(y_true)
                recall = 1.0 if np.sum(y_true) > 0 else 0.0
                specificity = 0.0 if np.sum(y_true == 0) > 0 else 1.0
            else:
                # All predicted negative
                precision = 0.0
                recall = 0.0
                specificity = 1.0 if np.sum(y_true == 0) > 0 else 0.0
        else:
            # Normal case
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Calculate specificity manually
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate other metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'mcc': float(mcc)
        }
    
    def _print_results(self):
        """Print optimization results"""
        
        print(f"\n🎯 THRESHOLD TUNING RESULTS")
        print(f"{'='*50}")
        
        # Dataset info
        dataset_info = self.results['dataset_info']
        print(f"Dataset: {dataset_info['total_samples']} samples")
        print(f"  {self.class_names[1]}: {dataset_info['positive_samples']} ({dataset_info['positive_samples']/dataset_info['total_samples']*100:.1f}%)")
        print(f"  {self.class_names[0]}: {dataset_info['negative_samples']} ({dataset_info['negative_samples']/dataset_info['total_samples']*100:.1f}%)")
        print(f"  Imbalance ratio: {dataset_info['class_imbalance_ratio']:.1f}:1")
        
        # Optimization results
        print(f"\nOptimization metric: {self.optimization_metric}")
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        
        # Performance comparison
        default_metrics = self.results['default_threshold_metrics']
        optimal_metrics = self.optimal_metrics
        
        print(f"\nPerformance Comparison:")
        print(f"{'Metric':<18} {'Default (0.5)':<12} {'Optimal':<12} {'Improvement':<12}")
        print(f"{'-'*54}")
        
        metrics_to_show = ['f1', 'precision', 'recall', 'specificity', 'balanced_accuracy', 'mcc']
        for metric in metrics_to_show:
            default_val = default_metrics[metric]
            optimal_val = optimal_metrics[metric]
            improvement = optimal_val - default_val
            improvement_str = f"{improvement:+.3f}" if abs(improvement) > 0.001 else "0.000"
            
            print(f"{metric.replace('_', ' ').title():<18} {default_val:<12.3f} {optimal_val:<12.3f} {improvement_str:<12}")
        
        # Highlight if significant improvement
        f1_improvement = optimal_metrics['f1'] - default_metrics['f1']
        if f1_improvement > 0.05:
            print(f"\n✅ Significant F1 improvement: +{f1_improvement:.3f}")
        elif f1_improvement > 0.01:
            print(f"\n🔹 Moderate F1 improvement: +{f1_improvement:.3f}")
        else:
            print(f"\n➖ Minimal F1 change: {f1_improvement:+.3f}")
    
    def create_threshold_plots(self, save_dir: str = None, filename_prefix: str = "threshold_tuning") -> str:
        """
        Create visualization plots for threshold tuning results
        
        Args:
            save_dir: Directory to save plots
            filename_prefix: Prefix for plot filenames
            
        Returns:
            Path to saved plot file
        """
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Threshold Tuning Results - Optimizing {self.optimization_metric.title()}', 
                     fontsize=16, fontweight='bold')
        
        sweep_data = self.results['threshold_sweep']
        thresholds = sweep_data['thresholds']
        
        # 1. Optimization metric vs threshold
        ax1 = axes[0, 0]
        ax1.plot(thresholds, sweep_data['optimization_values'], 'b-', linewidth=2, label=self.optimization_metric.title())
        ax1.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {self.optimal_threshold:.3f}')
        ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default: 0.5')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel(self.optimization_metric.title())
        ax1.set_title(f'{self.optimization_metric.title()} vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision and Recall vs threshold
        ax2 = axes[0, 1]
        ax2.plot(thresholds, sweep_data['precisions'], 'g-', linewidth=2, label='Precision')
        ax2.plot(thresholds, sweep_data['recalls'], 'r-', linewidth=2, label='Recall')
        ax2.plot(thresholds, sweep_data['f1_scores'], 'b-', linewidth=2, label='F1-Score')
        ax2.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision, Recall, F1 vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy metrics vs threshold
        ax3 = axes[1, 0]
        ax3.plot(thresholds, sweep_data['accuracies'], 'purple', linewidth=2, label='Accuracy')
        ax3.plot(thresholds, sweep_data['balanced_accuracies'], 'orange', linewidth=2, label='Balanced Accuracy')
        ax3.plot(thresholds, sweep_data['specificities'], 'brown', linewidth=2, label='Specificity')
        ax3.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Accuracy Metrics vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance comparison bar chart
        ax4 = axes[1, 1]
        metrics_names = ['F1', 'Precision', 'Recall', 'Specificity', 'Bal. Acc.']
        default_values = [
            self.results['default_threshold_metrics']['f1'],
            self.results['default_threshold_metrics']['precision'],
            self.results['default_threshold_metrics']['recall'],
            self.results['default_threshold_metrics']['specificity'],
            self.results['default_threshold_metrics']['balanced_accuracy']
        ]
        optimal_values = [
            self.optimal_metrics['f1'],
            self.optimal_metrics['precision'],
            self.optimal_metrics['recall'],
            self.optimal_metrics['specificity'],
            self.optimal_metrics['balanced_accuracy']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, default_values, width, label='Default (0.5)', alpha=0.7)
        bars2 = ax4.bar(x + width/2, optimal_values, width, label=f'Optimal ({self.optimal_threshold:.3f})', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if save_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"{filename_prefix}_{timestamp}.png"
            plot_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Threshold tuning plots saved to: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None
    
    def save_results(self, save_path: str):
        """Save threshold tuning results to JSON file"""
        
        # Add timestamp to results
        self.results['timestamp'] = datetime.now().isoformat()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save results
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Threshold tuning results saved to: {save_path}")
    
    def get_optimal_threshold(self) -> float:
        """Get the optimal threshold value"""
        return self.optimal_threshold
    
    def get_optimal_metrics(self) -> Dict:
        """Get metrics at optimal threshold"""
        return self.optimal_metrics
    
    def apply_optimal_threshold(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply optimal threshold to new probability predictions"""
        return (y_prob >= self.optimal_threshold).astype(int)

def tune_model_threshold(model, 
                        data_loader, 
                        device, 
                        optimization_metric: str = 'f1',
                        class_names: List[str] = None,
                        save_dir: str = None,
                        filename_prefix: str = "threshold_tuning") -> Dict:
    """
    Convenience function to tune threshold for a trained model
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for validation data
        device: Device to run inference on
        optimization_metric: Metric to optimize
        class_names: Names for the two classes
        save_dir: Directory to save results and plots
        filename_prefix: Prefix for saved files
        
    Returns:
        Threshold tuning results
    """
    
    print("🔧 Running threshold tuning on validation data...")
    
    # Get model predictions on validation data
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # Get model output
            output = model(data)
            
            # Apply softmax to get probabilities
            if output.shape[1] > 1:
                # Multi-class output, take probability of positive class
                probs = torch.softmax(output, dim=1)[:, 1]  # Probability of class 1 (cracked)
            else:
                # Single output, apply sigmoid
                probs = torch.sigmoid(output.squeeze())
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Initialize tuner and run optimization
    tuner = ThresholdTuner(
        optimization_metric=optimization_metric,
        class_names=class_names
    )
    
    results = tuner.tune_threshold(all_targets, all_probs)
    
    # Create plots and save results if requested
    if save_dir:
        # Save plots
        tuner.create_threshold_plots(save_dir, filename_prefix)
        
        # Save results
        results_filename = f"{filename_prefix}_results.json"
        results_path = os.path.join(save_dir, results_filename)
        tuner.save_results(results_path)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test threshold tuning module')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test with synthetic data')
    
    args = parser.parse_args()
    
    if args.test_synthetic:
        print("Testing threshold tuning with synthetic data...")
        
        # Generate synthetic imbalanced data
        np.random.seed(42)
        n_samples = 1000
        
        # Create imbalanced dataset (20% positive class)
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        # Generate probabilities with some class separation
        y_prob = np.random.beta(2, 5, n_samples)  # Skewed toward 0
        y_prob[y_true == 1] = np.random.beta(5, 2, np.sum(y_true))  # Higher probs for positive class
        
        # Test threshold tuning
        tuner = ThresholdTuner(optimization_metric='f1')
        results = tuner.tune_threshold(y_true, y_prob)
        
        # Create plots
        tuner.create_threshold_plots()
        
        print("\n✅ Threshold tuning test completed successfully!")
    else:
        print("Threshold tuning module loaded successfully!")
        print("Use --test-synthetic to run a test with synthetic data")