#!/usr/bin/env python3
"""
Manual Plotting Script for SDNET Spiking Neural Network
Comprehensive plotting capabilities for saved models and results
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
import pickle
from typing import Dict, List, Optional, Union

# Import from main modules
from sdnet_spiking import (
    SpikingResNetCrackDetector, SDNET2018Dataset, create_data_loaders,
    plot_training_history, plot_confusion_matrix, plot_model_comparison,
    plot_roc_curves, plot_training_time_comparison, plot_cross_validation_results
)
from spikingjelly.activation_based import neuron, surrogate
from baseline_models import CNNBaseline
from evaluation_metrics import ComprehensiveEvaluator

class ManualPlotter:
    """Comprehensive manual plotting tool for SDNET SNN project"""
    
    def __init__(self, output_dir='manual_plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluator = ComprehensiveEvaluator()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_model(self, model_path: str, model_type: str = 'snn', 
                   num_classes: int = 2, time_steps: int = 10) -> nn.Module:
        """Load a saved model"""
        print(f"Loading {model_type} model from {model_path}")
        
        if model_type.lower() == 'snn':
            model = SpikingResNetCrackDetector(
                spiking_neuron=neuron.IFNode,
                surrogate_function=surrogate.ATan(),
                num_classes=num_classes,
                T=time_steps
            )
            
            # Load checkpoint
            if model_path.endswith('.pth'):
                try:
                    # Try safe loading first
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                except Exception:
                    # Fall back to unsafe loading for compatibility
                    print("Warning: Using unsafe loading due to checkpoint format compatibility")
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Print additional info if available
                    if 'val_acc' in checkpoint:
                        val_acc = checkpoint['val_acc']
                        if isinstance(val_acc, (int, float)):
                            print(f"Model validation accuracy: {val_acc:.2f}%")
                    if 'model_type' in checkpoint:
                        print(f"Model type: {checkpoint['model_type']}")
                else:
                    model.load_state_dict(checkpoint)
            
        elif model_type.lower() in ['cnn', 'resnet50', 'resnet18', 'xception']:
            architecture = model_type.lower() if model_type.lower() != 'cnn' else 'resnet50'
            model = CNNBaseline(architecture, num_classes)
            
            # Load CNN model
            try:
                # Try safe loading first
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fall back to unsafe loading for compatibility
                print("Warning: Using unsafe loading due to checkpoint format compatibility")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Print additional info if available
                if 'val_acc' in checkpoint:
                    val_acc = checkpoint['val_acc']
                    if isinstance(val_acc, (int, float)):
                        print(f"Model validation accuracy: {val_acc:.2f}%")
                if 'model_name' in checkpoint:
                    print(f"Model name: {checkpoint['model_name']}")
            else:
                model.load_state_dict(checkpoint)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model.eval()
        print(f"Model loaded successfully!")
        return model
    
    def evaluate_model_on_data(self, model: nn.Module, data_loader: DataLoader, 
                              device: torch.device = torch.device('cpu')) -> Dict:
        """Evaluate a model and return comprehensive metrics"""
        model.to(device)
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                pred = output.argmax(dim=1)
                prob = torch.softmax(output, dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx}/{len(data_loader)}")
        
        # Calculate comprehensive metrics
        metrics = self.evaluator.calculate_comprehensive_metrics(
            np.array(all_targets), 
            np.array(all_preds),
            np.array(all_probs)
        )
        
        # Add raw data for plotting
        metrics['y_true'] = np.array(all_targets)
        metrics['y_pred'] = np.array(all_preds)
        metrics['y_prob'] = np.array(all_probs)
        
        return metrics
    
    def plot_model_analysis(self, model: nn.Module, data_loader: DataLoader,
                           model_name: str = 'model', device: torch.device = torch.device('cpu')):
        """Complete analysis and plotting for a single model"""
        print(f"\n{'='*60}")
        print(f"ANALYZING MODEL: {model_name}")
        print('='*60)
        
        # Evaluate model
        metrics = self.evaluate_model_on_data(model, data_loader, device)
        
        # Print metrics
        print(f"\nModel Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Generate plots
        model_dir = os.path.join(self.output_dir, model_name.replace(' ', '_').lower())
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plot_confusion_matrix(
            metrics['y_true'], metrics['y_pred'], 
            save_dir=model_dir, model_name=model_name
        )
        
        # 2. ROC Curve
        self.plot_single_roc_curve(metrics, model_name, model_dir)
        
        # 3. Class Distribution
        self.plot_class_distribution(metrics['y_true'], metrics['y_pred'], model_name, model_dir)
        
        # 4. Prediction Confidence Distribution
        self.plot_confidence_distribution(metrics['y_prob'], metrics['y_true'], model_name, model_dir)
        
        return metrics
    
    def plot_single_roc_curve(self, metrics: Dict, model_name: str, save_dir: str):
        """Plot ROC curve for a single model"""
        if 'y_true' not in metrics or 'y_prob' not in metrics:
            return
            
        y_true = metrics['y_true']
        y_prob = metrics['y_prob'][:, 1]  # Probability of positive class
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, save_dir: str):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(['Uncracked', 'Cracked'], counts_true, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax1.set_title('True Class Distribution')
        ax1.set_ylabel('Count')
        for i, count in enumerate(counts_true):
            ax1.text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(['Uncracked', 'Cracked'], counts_pred, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax2.set_title('Predicted Class Distribution')
        ax2.set_ylabel('Count')
        for i, count in enumerate(counts_pred):
            ax2.text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        plt.suptitle(f'Class Distribution - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'class_distribution_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class distribution plot saved to {save_path}")
    
    def plot_confidence_distribution(self, y_prob: np.ndarray, y_true: np.ndarray, 
                                   model_name: str, save_dir: str):
        """Plot prediction confidence distribution"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        max_probs = np.max(y_prob, axis=1)
        ax1.hist(max_probs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Maximum Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Confidence by class
        correct_mask = (np.argmax(y_prob, axis=1) == y_true)
        correct_conf = max_probs[correct_mask]
        incorrect_conf = max_probs[~correct_mask]
        
        ax2.hist([correct_conf, incorrect_conf], bins=30, alpha=0.7, 
                label=['Correct', 'Incorrect'], color=['green', 'red'])
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence by Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confidence for each class
        class_0_mask = (y_true == 0)
        class_1_mask = (y_true == 1)
        
        ax3.hist([max_probs[class_0_mask], max_probs[class_1_mask]], 
                bins=30, alpha=0.7, label=['Uncracked', 'Cracked'], 
                color=['skyblue', 'lightcoral'])
        ax3.set_xlabel('Prediction Confidence')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Confidence by True Class')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calibration plot (reliability diagram)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_mask[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
        ax4.plot(confidences, accuracies, 'o-', color='red', label='Model')
        ax4.set_xlabel('Mean Predicted Confidence')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Calibration Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Prediction Confidence Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'confidence_analysis_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence analysis plot saved to {save_path}")
    
    def compare_multiple_models(self, model_configs: List[Dict], data_loader: DataLoader,
                               device: torch.device = torch.device('cpu')):
        """Compare multiple models and generate comparison plots"""
        print(f"\n{'='*60}")
        print("COMPARING MULTIPLE MODELS")
        print('='*60)
        
        all_results = {}
        
        # Evaluate each model
        for config in model_configs:
            model_name = config['name']
            model_path = config['path']
            model_type = config.get('type', 'snn')
            
            print(f"\nEvaluating {model_name}...")
            
            # Load and evaluate model
            model = self.load_model(model_path, model_type)
            metrics = self.evaluate_model_on_data(model, data_loader, device)
            all_results[model_name] = metrics
        
        # Generate comparison plots
        print(f"\nGenerating comparison plots...")
        
        # 1. Model comparison chart
        plot_model_comparison(all_results, save_dir=self.output_dir)
        
        # 2. ROC curves comparison
        plot_roc_curves(all_results, save_dir=self.output_dir)
        
        # 3. Individual confusion matrices
        for model_name, metrics in all_results.items():
            plot_confusion_matrix(
                metrics['y_true'], metrics['y_pred'],
                save_dir=self.output_dir, model_name=model_name
            )
        
        # 4. Performance summary table
        self.generate_performance_table(all_results)
        
        return all_results
    
    def generate_performance_table(self, results: Dict):
        """Generate and save performance comparison table"""
        import pandas as pd
        
        # Extract metrics
        metrics_data = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
            }
            if 'auc_roc' in metrics:
                row['AUC-ROC'] = f"{metrics['auc_roc']:.4f}"
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'performance_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"Performance table saved to {csv_path}")
        
        # Display table
        print("\nPerformance Comparison:")
        print(df.to_string(index=False))
    
    def plot_from_saved_results(self, results_path: str):
        """Load and plot from saved JSON results"""
        print(f"Loading results from {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Handle different result formats
        if 'baseline_comparison' in results:
            # Comprehensive evaluation results
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                plot_cross_validation_results(cv_results, save_dir=self.output_dir)
            
            baseline_results = results['baseline_comparison']
            plot_model_comparison(baseline_results, save_dir=self.output_dir)
            plot_training_time_comparison(baseline_results, save_dir=self.output_dir)
            
        elif 'fold_results' in results:
            # Cross-validation results
            plot_cross_validation_results(results, save_dir=self.output_dir)
            
        else:
            # Baseline comparison results
            plot_model_comparison(results, save_dir=self.output_dir)
            if any('training_time' in model_results for model_results in results.values()):
                plot_training_time_comparison(results, save_dir=self.output_dir)
        
        print(f"Plots generated in {self.output_dir}")
    
    def plot_training_history_from_file(self, history_path: str, model_name: str = 'model'):
        """Plot training history from saved training data"""
        if history_path.endswith('.json'):
            with open(history_path, 'r') as f:
                history = json.load(f)
        elif history_path.endswith('.pkl'):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")
        
        # Extract training data
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        train_accs = history.get('train_accs', [])
        val_accs = history.get('val_accs', [])
        
        if train_losses and val_losses:
            model_dir = os.path.join(self.output_dir, f"{model_name}_training")
            os.makedirs(model_dir, exist_ok=True)
            plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir=model_dir)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Manual plotting tool for SDNET SNN models')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'compare', 'results', 'history'],
                       help='Plotting mode')
    
    # Single model analysis
    parser.add_argument('--model-path', type=str, help='Path to saved model')
    parser.add_argument('--model-type', type=str, default='snn',
                       choices=['snn', 'cnn', 'resnet50', 'resnet18', 'xception'],
                       help='Type of model')
    parser.add_argument('--model-name', type=str, default='model', help='Name for the model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./SDNET2018',
                       help='Path to SDNET2018 dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--time-steps', type=int, default=10, help='Time steps for SNN')
    
    # Multiple model comparison
    parser.add_argument('--config-file', type=str, 
                       help='JSON config file for multiple model comparison')
    
    # Results plotting
    parser.add_argument('--results-file', type=str,
                       help='Path to saved results JSON file')
    
    # Training history
    parser.add_argument('--history-file', type=str,
                       help='Path to saved training history file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='manual_plots',
                       help='Directory to save plots')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize plotter
    plotter = ManualPlotter(args.output_dir)
    
    if args.mode == 'single':
        # Single model analysis
        if not args.model_path:
            raise ValueError("--model-path required for single model mode")
        
        # Create data loader
        _, val_loader = create_data_loaders(
            args.data_dir, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=0.8
        )
        
        # Load and analyze model
        model = plotter.load_model(args.model_path, args.model_type, 
                                  args.num_classes, args.time_steps)
        plotter.plot_model_analysis(model, val_loader, args.model_name, device)
        
    elif args.mode == 'compare':
        # Multiple model comparison
        if not args.config_file:
            raise ValueError("--config-file required for compare mode")
        
        with open(args.config_file, 'r') as f:
            model_configs = json.load(f)
        
        # Create data loader
        _, val_loader = create_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=0.8
        )
        
        plotter.compare_multiple_models(model_configs, val_loader, device)
        
    elif args.mode == 'results':
        # Plot from saved results
        if not args.results_file:
            raise ValueError("--results-file required for results mode")
        
        plotter.plot_from_saved_results(args.results_file)
        
    elif args.mode == 'history':
        # Plot training history
        if not args.history_file:
            raise ValueError("--history-file required for history mode")
        
        plotter.plot_training_history_from_file(args.history_file, args.model_name)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()