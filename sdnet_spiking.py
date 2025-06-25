"""
Spiking ResNet Implementation for SDNET2018 Concrete Crack Detection
Using SpikingJelly Framework

Dataset: SDNET2018 - 56,000+ annotated concrete crack images (256x256)
Framework: SpikingJelly with PyTorch backend
Architecture: Spiking ResNet-18 adapted for binary classification
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import new evaluation modules
from evaluation_metrics import ComprehensiveEvaluator, create_crackvision_style_table
from cross_validation import CrossValidator, run_cross_validation_experiment
from baseline_models import CNNBaseline, TraditionalBaseline, ModelComparator

# SpikingJelly imports
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model import spiking_resnet

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SDNET2018Dataset(Dataset):
    """
    Custom Dataset for SDNET2018 concrete crack images
    
    Directory structure expected:
    root/
        D/ (Decks)
            CD/ (Cracked)
            UD/ (Uncracked)
        P/ (Pavements)
            CP/
            UP/
        W/ (Walls)
            CW/
            UW/
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None, train_ratio: float = 0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Define subdirectories
        subdirs = {
            'D': ['CD', 'UD'],  # Decks
            'P': ['CP', 'UP'],  # Pavements
            'W': ['CW', 'UW']   # Walls
        }
        
        # Load all image paths and labels
        for main_dir, sub_list in subdirs.items():
            for sub_dir in sub_list:
                # Label: 1 for cracked (C*), 0 for uncracked (U*)
                label = 1 if sub_dir.startswith('C') else 0
                
                dir_path = os.path.join(root_dir, main_dir, sub_dir)
                if os.path.exists(dir_path):
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)
        
        # Split dataset
        total_images = len(self.images)
        indices = np.arange(total_images)
        np.random.shuffle(indices)
        
        train_size = int(total_images * train_ratio)
        
        if split == 'train':
            indices = indices[:train_size]
        else:  # validation
            indices = indices[train_size:]
        
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        print(f"{split} dataset: {len(self.images)} images")
        print(f"Cracked: {sum(self.labels)}, Uncracked: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SpikingResNetCrackDetector(nn.Module):
    """
    Spiking ResNet for crack detection
    Modified from SpikingJelly's implementation for binary classification
    """
    def __init__(self, 
                 spiking_neuron: callable = neuron.IFNode,
                 surrogate_function: callable = surrogate.ATan(),
                 detach_reset: bool = True,
                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 T: int = 4):  # Time steps
        super().__init__()
        
        self.T = T
        
        # Use SpikingJelly's pre-built spiking ResNet
        # Modify for binary classification
        self.model = spiking_resnet.spiking_resnet18(
            pretrained=False,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual
        )
        
    def forward(self, x):
        # x shape: [N, C, H, W]
        # Repeat input for T timesteps
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]
        
        # Process through time steps
        out_spikes = []
        for t in range(self.T):
            out = self.model(x_seq[t])
            out_spikes.append(out)
        
        # Aggregate spikes over time
        out = torch.stack(out_spikes, dim=0).mean(dim=0)  # [N, num_classes]
        
        # Reset the network state
        functional.reset_net(self.model)
        
        return out

class DirectEncodingSpikingResNet(nn.Module):
    """
    Alternative: Direct encoding approach without temporal repetition
    Uses rate coding implicitly through the first convolutional layer
    """
    def __init__(self,
                 num_classes: int = 2,
                 T: int = 4):
        super().__init__()
        
        self.T = T
        
        # Encoder: Convert image to spikes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        
        # Main spiking ResNet body (without the first conv layer)
        self.spiking_resnet = spiking_resnet.spiking_resnet18(
            pretrained=False,
            num_classes=num_classes
        )
        
        # Remove the first conv layer from resnet
        self.features = nn.Sequential(*list(self.spiking_resnet.children())[1:])
        
    def forward(self, x):
        # Direct encoding through first layer
        spike_input = self.encoder(x)
        
        # Process through ResNet
        out_spikes = []
        for _ in range(self.T):
            out = self.features(spike_input)
            out_spikes.append(out)
        
        # Aggregate output
        out = torch.stack(out_spikes, dim=0).mean(dim=0)
        
        functional.reset_net(self)
        
        return out

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, train_ratio: float = 0.8):
    """Create data loaders with appropriate transforms"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SDNET2018Dataset(data_dir, split='train', transform=train_transform, train_ratio=train_ratio)
    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform, train_ratio=train_ratio)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir='plots'):
    """Plot training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_dir='plots', model_name='model'):
    """Plot confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Uncracked', 'Cracked'],
                yticklabels=['Uncracked', 'Cracked'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return save_path

def plot_model_comparison(results_dict, save_dir='plots'):
    """Plot comprehensive model comparison charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Extract metric values
    data = {metric: [] for metric in metrics}
    for model in models:
        for metric in metrics:
            value = results_dict[model].get(metric, 0)
            # Convert to percentage if needed
            if metric == 'accuracy' and value <= 1.0:
                value *= 100
            elif metric in ['precision', 'recall', 'f1_score'] and value <= 1.0:
                value *= 100
            data[metric].append(value)
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    axes = [ax1, ax2, ax3, ax4]
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(models, data[metric], color=colors, alpha=0.8, edgecolor='black')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(data[metric]) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {save_path}")
    return save_path

def plot_roc_curves(models_data, save_dir='plots'):
    """Plot ROC curves for multiple models"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))
    
    for i, (model_name, data) in enumerate(models_data.items()):
        if 'auc_roc' in data and 'y_true' in data and 'y_prob' in data:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'][:, 1])
            auc_score = data['auc_roc']
            plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves comparison saved to {save_path}")
    return save_path

def plot_training_time_comparison(results_dict, save_dir='plots'):
    """Plot training time and efficiency comparisons"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results_dict.keys())
    training_times = []
    inference_times = []
    model_sizes = []
    
    for model in models:
        training_times.append(results_dict[model].get('training_time', 0))
        inference_times.append(results_dict[model].get('inference_time_per_image', 0) * 1000)  # Convert to ms
        model_sizes.append(results_dict[model].get('model_size', 0))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # Training time
    bars1 = ax1.bar(models, training_times, color=colors, alpha=0.8)
    ax1.set_title('Training Time Comparison', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.grid(True, alpha=0.3)
    for bar, time in zip(bars1, training_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Inference time
    bars2 = ax2.bar(models, inference_times, color=colors, alpha=0.8)
    ax2.set_title('Inference Time per Image', fontweight='bold')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.grid(True, alpha=0.3)
    for bar, time in zip(bars2, inference_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Model size
    bars3 = ax3.bar(models, model_sizes, color=colors, alpha=0.8)
    ax3.set_title('Model Size Comparison', fontweight='bold')
    ax3.set_ylabel('Model Size (MB)')
    ax3.grid(True, alpha=0.3)
    for bar, size in zip(bars3, model_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency scatter plot (Accuracy vs Training Time)
    accuracies = [results_dict[model].get('accuracy', 0) * 100 for model in models]
    scatter = ax4.scatter(training_times, accuracies, c=range(len(models)), 
                         cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Efficiency: Accuracy vs Training Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add model labels to scatter plot
    for i, model in enumerate(models):
        ax4.annotate(model, (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Rotate x-axis labels if needed
    for ax in [ax1, ax2, ax3]:
        if len(max(models, key=len)) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'efficiency_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Efficiency comparison plot saved to {save_path}")
    return save_path

def plot_cross_validation_results(cv_results, save_dir='plots'):
    """Plot cross-validation results with error bars"""
    os.makedirs(save_dir, exist_ok=True)
    
    if 'fold_results' not in cv_results:
        return None
    
    # Extract metrics across folds
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    fold_data = {metric: [] for metric in metrics}
    
    for fold_result in cv_results['fold_results']:
        for metric in metrics:
            if metric in fold_result:
                fold_data[metric].append(fold_result[metric] * 100 if fold_result[metric] <= 1.0 else fold_result[metric])
    
    # Calculate mean and std
    means = [np.mean(fold_data[metric]) for metric in metrics]
    stds = [np.std(fold_data[metric]) for metric in metrics]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(metrics))
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Cross-Validation Results (Mean ± Std)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Box plot for distribution
    box_data = [fold_data[metric] for metric in metrics]
    bp = ax2.boxplot(box_data, labels=[m.replace('_', ' ').title() for m in metrics],
                     patch_artist=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Cross-Validation Score Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cross_validation_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-validation results plot saved to {save_path}")
    return save_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Spiking ResNet for SDNET2018 Concrete Crack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, 
                       default='/home/duyanh/Workspace/SDNET_spiking/SDNET2018',
                       help='Path to SDNET2018 dataset directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    
    # Model parameters
    parser.add_argument('--time-steps', '-T', type=int, default=10,
                       help='Number of time steps for spiking network')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of output classes')
    
    # Training options
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision training')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Checkpoint and output
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Data augmentation
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (rest for validation)')
    
    # Evaluation modes (Phase 1 additions)
    parser.add_argument('--eval-mode', type=str, default='train',
                       choices=['train', 'cross_validation', 'baseline_comparison', 'comprehensive'],
                       help='Evaluation mode to run')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--compare-baselines', action='store_true',
                       help='Compare with CNN baselines (ResNet50, etc.)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio
    )
    
    # Create model
    model = SpikingResNetCrackDetector(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=args.num_classes,
        T=args.time_steps
    ).to(device)
    
    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    
    # Mixed precision scaler
    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    print(f"Mixed precision training: {use_amp}")
    
    # Handle different evaluation modes
    if args.eval_mode == 'cross_validation':
        return run_cross_validation_mode(args, device)
    elif args.eval_mode == 'baseline_comparison':
        return run_baseline_comparison_mode(args, device)
    elif args.eval_mode == 'comprehensive':
        return run_comprehensive_evaluation_mode(args, device)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    # Training loop  
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler
        )
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args,
                'model_type': 'snn'
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    # Create plots directory
    plots_dir = os.path.join(args.save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir=plots_dir)
    
    # Final evaluation with best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, final_preds, final_targets = evaluate(
        model, val_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_targets, final_preds, 
                              target_names=['Uncracked', 'Cracked']))
    
    # Confusion matrix
    plot_confusion_matrix(final_targets, final_preds, save_dir=plots_dir, model_name='Spiking_ResNet')
    
    # Save final model (consistent format with best_model.pth)
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_val_acc': best_val_acc,
        'args': args,
        'model_type': 'snn'
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    print(f"Final model saved to {os.path.join(args.save_dir, 'final_model.pth')}")

def inference_example():
    """Example inference function for a single image"""
    # Load model
    model = SpikingResNetCrackDetector(num_classes=2, T=4)
    model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and predict
    image_path = 'path/to/test/image.jpg'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
    
    labels = ['Uncracked', 'Cracked']
    print(f'Prediction: {labels[pred]}')
    print(f'Confidence: {prob[0, pred].item():.2%}')

def run_cross_validation_mode(args, device):
    """Run cross-validation evaluation mode"""
    print("\n" + "="*60)
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    # Create full dataset for CV
    full_dataset = SDNET2018Dataset(
        args.data_dir, 
        split='train',  # We'll handle splitting in CV
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        train_ratio=1.0  # Use full dataset
    )
    
    # CV configuration
    config = {
        'n_folds': args.cv_folds,
        'train_params': {
            'model_params': {
                'spiking_neuron': neuron.IFNode,
                'surrogate_function': surrogate.ATan(),
                'num_classes': args.num_classes,
                'T': args.time_steps
            },
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_epochs': args.num_epochs
        },
        'data_params': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        },
        'save_path': os.path.join(args.results_dir, 'snn_cv_results.json')
    }
    
    # Run CV evaluation
    results = run_cross_validation_experiment(
        SpikingResNetCrackDetector, full_dataset, config
    )
    
    # Generate cross-validation plots
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating cross-validation plots...")
    plot_cross_validation_results(results, save_dir=plots_dir)
    
    print("\nCross-validation completed successfully!")
    return results

def run_baseline_comparison_mode(args, device):
    """Run baseline comparison evaluation mode"""
    print("\n" + "="*60)
    print("RUNNING BASELINE COMPARISON EVALUATION")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio
    )
    
    comparator = ModelComparator()
    
    # Evaluate SNN
    print("\n1. Evaluating Spiking ResNet...")
    snn_model = SpikingResNetCrackDetector(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=args.num_classes,
        T=args.time_steps
    ).to(device)
    
    snn_results = evaluate_model_comprehensive(snn_model, train_loader, val_loader, device, args)
    comparator.add_model_results("Spiking ResNet", snn_results)
    
    # Save the SNN model from baseline comparison
    save_baseline_model(snn_model, "Spiking ResNet", snn_results, save_dir=args.save_dir)
    
    # Evaluate CNN baselines
    print("\n2. Evaluating CNN Baselines...")
    cnn_models = {
        "ResNet50 (CrackVision)": CNNBaseline('resnet50', args.num_classes),
        "ResNet18": CNNBaseline('resnet18', args.num_classes),
        "Xception-style": CNNBaseline('xception_style', args.num_classes)
    }
    
    for model_name, model in cnn_models.items():
        print(f"\nEvaluating {model_name}...")
        model = model.to(device)
        results = evaluate_model_comprehensive(model, train_loader, val_loader, device, args)
        comparator.add_model_results(model_name, results)
        
        # Save the trained model
        save_baseline_model(model, model_name, results, save_dir=args.save_dir)
    
    # Generate comparison table
    comparator.generate_crackvision_comparison_table()
    
    # Generate comprehensive figures
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating comparison plots...")
    plot_model_comparison(comparator.results, save_dir=plots_dir)
    plot_training_time_comparison(comparator.results, save_dir=plots_dir)
    
    # Generate individual confusion matrices for each model
    for model_name in comparator.results.keys():
        if 'y_true' in comparator.results[model_name] and 'y_pred' in comparator.results[model_name]:
            plot_confusion_matrix(
                comparator.results[model_name]['y_true'], 
                comparator.results[model_name]['y_pred'],
                save_dir=plots_dir, 
                model_name=model_name
            )
    
    # Save results
    results_path = os.path.join(args.results_dir, 'baseline_comparison.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(comparator.results, f, indent=2, default=str)
    
    print(f"\nBaseline comparison results saved to {results_path}")
    print(f"Plots saved to {plots_dir}")
    return comparator.results

def run_baseline_comparison_cnn_only(args, device):
    """Run baseline comparison evaluation mode for CNN models only (no SNN)"""
    print("\n" + "="*50)
    print("RUNNING CNN BASELINE EVALUATION ONLY")
    print("="*50)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio
    )
    
    comparator = ModelComparator()
    
    # Evaluate CNN baselines only
    print("\nEvaluating CNN Baselines...")
    cnn_models = {
        "ResNet50 (CrackVision)": CNNBaseline('resnet50', args.num_classes),
        "ResNet18": CNNBaseline('resnet18', args.num_classes),
        "Xception-style": CNNBaseline('xception_style', args.num_classes)
    }
    
    for model_name, model in cnn_models.items():
        print(f"\nEvaluating {model_name}...")
        model = model.to(device)
        results = evaluate_model_comprehensive(model, train_loader, val_loader, device, args)
        comparator.add_model_results(model_name, results)
        
        # Save the trained model
        save_baseline_model(model, model_name, results, save_dir=args.save_dir)
    
    # Generate comparison table
    comparator.generate_crackvision_comparison_table()
    
    # Generate figures for CNN models
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating CNN comparison plots...")
    plot_model_comparison(comparator.results, save_dir=plots_dir)
    plot_training_time_comparison(comparator.results, save_dir=plots_dir)
    
    return comparator.results

def run_comprehensive_evaluation_mode(args, device):
    """Run comprehensive evaluation combining CV and baseline comparison"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Run cross-validation first (this trains SNN multiple times across folds)
    print("\nPhase 1: Cross-Validation...")
    cv_results = run_cross_validation_mode(args, device)
    
    # Run baseline comparison WITHOUT training SNN again 
    # (only train CNN baselines since SNN was already evaluated in CV)
    print("\nPhase 2: Baseline Comparison (CNN models only)...")
    baseline_results = run_baseline_comparison_cnn_only(args, device)
    
    # Add SNN results from cross-validation to baseline comparison
    if 'mean_metrics' in cv_results:
        snn_cv_metrics = cv_results['mean_metrics']
        # Convert CV metrics to baseline comparison format
        snn_baseline_metrics = {
            'accuracy': snn_cv_metrics.get('accuracy', 0),
            'precision': snn_cv_metrics.get('precision', 0),
            'recall': snn_cv_metrics.get('recall', 0),
            'f1_score': snn_cv_metrics.get('f1_score', 0),
            'auc_roc': snn_cv_metrics.get('auc_roc', 0),
            'training_time': snn_cv_metrics.get('total_training_time', 0) / args.cv_folds,
            'inference_time': snn_cv_metrics.get('inference_time', 0),
            'total_parameters': snn_cv_metrics.get('total_parameters', 0),
            'model_size': snn_cv_metrics.get('model_size', 0)
        }
        baseline_results['Spiking ResNet (CV)'] = snn_baseline_metrics
    
    # Combined analysis
    print("\nPhase 3: Combined Analysis and Figure Generation...")
    
    # Create comprehensive comparison table
    create_crackvision_style_table(baseline_results, "SDNET2018")
    
    # Generate comprehensive figures
    plots_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating comprehensive comparison plots...")
    plot_model_comparison(baseline_results, save_dir=plots_dir)
    plot_training_time_comparison(baseline_results, save_dir=plots_dir)
    
    # Generate cross-validation plots if available
    if cv_results and 'fold_results' in cv_results:
        plot_cross_validation_results(cv_results, save_dir=plots_dir)
    
    # Save comprehensive results
    from datetime import datetime
    comprehensive_results = {
        'cross_validation': cv_results,
        'baseline_comparison': baseline_results,
        'evaluation_date': str(datetime.now()),
        'configuration': vars(args)
    }
    
    results_path = os.path.join(args.results_dir, 'comprehensive_evaluation.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\nComprehensive evaluation results saved to {results_path}")
    return comprehensive_results

def evaluate_model_comprehensive(model, train_loader, val_loader, device, args):
    """Comprehensive model evaluation following CrackVision methodology"""
    import time
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=min(args.num_epochs, 10)
    )
    
    # Mixed precision scaler
    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    start_time = time.time()
    best_val_acc = 0
    
    # Proper training loop (using main function logic)
    for epoch in range(min(args.num_epochs, 10)):  # Limit epochs for comparison
        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler
        )
        
        # Evaluate
        _, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 2 == 0:  # Print every 2 epochs to reduce output
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            prob = torch.softmax(output, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.calculate_comprehensive_metrics(
        np.array(all_targets), 
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Add timing information
    metrics['training_time'] = training_time
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_image'] = inference_time / len(all_targets)
    
    # Add model size (approximate)
    total_params = sum(p.numel() for p in model.parameters())
    metrics['total_parameters'] = total_params
    metrics['model_size'] = total_params * 4 / (1024 * 1024)  # MB (assuming float32)
    
    # Store data for figure generation
    metrics['y_true'] = np.array(all_targets)
    metrics['y_pred'] = np.array(all_preds)
    metrics['y_prob'] = np.array(all_probs)
    
    return metrics

def save_baseline_model(model, model_name: str, metrics: dict, save_dir: str = 'checkpoints'):
    """Save a baseline model with consistent naming and metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create safe filename
    safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    model_path = os.path.join(save_dir, f'{safe_name}_baseline.pth')
    
    # Determine model type
    model_type = 'cnn'
    if hasattr(model, 'architecture'):
        model_type = model.architecture
    elif 'resnet50' in model_name.lower():
        model_type = 'resnet50'
    elif 'resnet18' in model_name.lower():
        model_type = 'resnet18'
    elif 'xception' in model_name.lower():
        model_type = 'xception'
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'model_type': model_type,
        'val_acc': metrics.get('accuracy', 0),
        'metrics': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    }, model_path)
    
    print(f"Saved {model_name} to {model_path}")
    return model_path

if __name__ == '__main__':
    main()
    # For inference on new images:
    # inference_example()