#!/usr/bin/env python3
"""
Spiking ResNet Training for SDNET2018 Concrete Crack Detection
Using SpikingJelly Framework

Focused on training and evaluation of Spiking Neural Networks only.
Separated from baseline comparisons and plotting functionality.

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
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Import evaluation modules
from evaluation_metrics import ComprehensiveEvaluator
from cross_validation import CrossValidator, run_cross_validation_experiment

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
    all_probs = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Store predictions and probabilities for comprehensive evaluation
            prob = torch.softmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)

def comprehensive_evaluate(model, loader, device):
    """Perform comprehensive evaluation with all metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    inference_start = time.time()
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            prob = torch.softmax(output, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    inference_time = time.time() - inference_start
    
    # Calculate comprehensive metrics
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.calculate_comprehensive_metrics(
        np.array(all_targets), 
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Add timing information
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_image'] = inference_time / len(all_targets)
    
    # Store data for analysis
    metrics['y_true'] = np.array(all_targets)
    metrics['y_pred'] = np.array(all_preds)
    metrics['y_prob'] = np.array(all_probs)
    
    return metrics

def save_model_checkpoint(model, optimizer, epoch, val_acc, args, model_name, save_dir, add_timestamp=True):
    """Save model checkpoint with metadata and optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args),
        'model_type': 'snn',
        'model_name': model_name,
        'save_time': str(datetime.now()),
        'timestamp': timestamp
    }
    
    # Create filename with timestamp
    if timestamp:
        filename = f'{model_name}_{timestamp}.pth'
    else:
        filename = f'{model_name}.pth'
    
    model_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, model_path)
    print(f'Saved {model_name} to {model_path}')
    
    # Also save as latest (without timestamp) for easy access
    if add_timestamp:
        latest_path = os.path.join(save_dir, f'{model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        print(f'Also saved as latest: {latest_path}')
    
    return model_path

def save_evaluation_results(results, save_dir, filename, add_timestamp=True):
    """Save evaluation results to JSON file with optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        else:
            json_results[key] = value
    
    # Add metadata to results
    json_results['save_metadata'] = {
        'save_time': str(datetime.now()),
        'timestamp': timestamp,
        'original_filename': filename
    }
    
    # Create filename with timestamp
    if timestamp and add_timestamp:
        base_name = filename.replace('.json', '')
        timestamped_filename = f'{base_name}_{timestamp}.json'
    else:
        timestamped_filename = filename
    
    results_path = os.path.join(save_dir, timestamped_filename)
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to {results_path}")
    
    # Also save as latest (without timestamp) for easy access
    if add_timestamp:
        base_name = filename.replace('.json', '')
        latest_path = os.path.join(save_dir, f'{base_name}_latest.json')
        with open(latest_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"Also saved as latest: {latest_path}")
    
    return results_path

def print_comprehensive_results(metrics, model_name="Spiking ResNet"):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION RESULTS - {model_name}")
    print(f"{'='*60}")
    
    # Core metrics
    print(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:    {metrics.get('precision', 0):.4f}")
    print(f"Recall:       {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:     {metrics.get('f1_score', 0):.4f}")
    print(f"Specificity:  {metrics.get('specificity', 0):.4f}")
    print(f"AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")
    
    # Performance metrics
    if 'training_time' in metrics:
        print(f"\nTraining Time:      {metrics['training_time']:.2f} seconds")
    if 'inference_time' in metrics:
        print(f"Inference Time:     {metrics['inference_time']:.4f} seconds")
        print(f"Per Image:          {metrics['inference_time_per_image']*1000:.2f} ms")
    
    # Model size
    if 'total_parameters' in metrics:
        print(f"\nModel Parameters:   {metrics['total_parameters']:,}")
        print(f"Model Size:         {metrics.get('model_size', 0):.2f} MB")
    
    print(f"{'='*60}")

def run_cross_validation(args, device):
    """Run cross-validation evaluation"""
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
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Spiking ResNet Training for SDNET2018 Concrete Crack Detection',
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
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Data splitting
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (rest for validation)')
    
    # Evaluation modes
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cross_validation', 'comprehensive'],
                       help='Execution mode')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    # Evaluation only (no training)
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate pre-trained model, skip training')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model for evaluation')
    
    # Output options
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Disable timestamps in output filenames')
    
    return parser.parse_args()

def create_timestamped_directories(base_save_dir: str, base_results_dir: str, use_timestamps: bool = True):
    """Create timestamped directories for organizing outputs"""
    if use_timestamps:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"snn_run_{timestamp}"
        
        save_dir = os.path.join(base_save_dir, run_name)
        results_dir = os.path.join(base_results_dir, run_name)
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Created timestamped directories:")
        print(f"  Checkpoints: {save_dir}")
        print(f"  Results: {results_dir}")
        
        return save_dir, results_dir, run_name
    else:
        # Use original directories without timestamping
        os.makedirs(base_save_dir, exist_ok=True)
        os.makedirs(base_results_dir, exist_ok=True)
        return base_save_dir, base_results_dir, None

def main():
    """Main training and evaluation function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Determine timestamp usage
    use_timestamps = not args.no_timestamps
    print(f"Using timestamps in folders: {use_timestamps}")
    
    # Create timestamped directories
    save_dir, results_dir, run_name = create_timestamped_directories(
        args.save_dir, args.results_dir, use_timestamps
    )
    
    # Update args with new directories
    args.save_dir = save_dir
    args.results_dir = results_dir
    
    # Handle different modes
    if args.mode == 'cross_validation':
        print("\nRunning Cross-Validation Mode...")
        results = run_cross_validation(args, device)
        
        # Save CV results
        save_evaluation_results(
            results, 
            args.results_dir, 
            'spiking_resnet_cv_results.json',
            add_timestamp=False  # Folder already timestamped
        )
        
        print("\nCross-validation completed successfully!")
        return results
    
    elif args.mode == 'comprehensive':
        print("\nRunning Comprehensive Evaluation Mode...")
        
        # First run regular training
        print("Phase 1: Standard Training...")
        training_results = main_training_loop(args, device, use_timestamps, run_name)
        
        # Then run cross-validation
        print("Phase 2: Cross-Validation...")
        cv_results = run_cross_validation(args, device)
        
        # Combine results
        comprehensive_results = {
            'standard_training': training_results,
            'cross_validation': cv_results,
            'evaluation_date': str(datetime.now()),
            'configuration': vars(args)
        }
        
        # Save comprehensive results
        save_evaluation_results(
            comprehensive_results,
            args.results_dir,
            'spiking_resnet_comprehensive.json',
            add_timestamp=False  # Folder already timestamped
        )
        
        print("\nComprehensive evaluation completed successfully!")
        return comprehensive_results
    
    else:
        # Standard training mode
        return main_training_loop(args, device, use_timestamps, run_name)

def main_training_loop(args, device, use_timestamps=True, run_name=None):
    """Main training loop for standard mode"""
    
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
    
    # Model size information
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Handle evaluation-only mode
    if args.eval_only:
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading pre-trained model from {args.model_path}")
            try:
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
            except Exception:
                print("Warning: Using unsafe loading due to checkpoint format compatibility")
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error: --eval-only requires --model-path to be specified and valid")
            return None
        
        # Perform comprehensive evaluation
        print("Performing comprehensive evaluation...")
        metrics = comprehensive_evaluate(model, val_loader, device)
        
        # Add model info
        metrics['total_parameters'] = total_params
        metrics['model_size'] = model_size_mb
        
        # Print and save results
        print_comprehensive_results(metrics, "Spiking ResNet (Pre-trained)")
        save_evaluation_results(metrics, args.results_dir, 'spiking_resnet_eval_only.json', add_timestamp=False)
        
        return metrics
    
    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            print("Warning: Using unsafe loading due to checkpoint format compatibility")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
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
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    training_start_time = time.time()
    
    # Training loop  
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler
        )
        
        # Evaluate
        val_loss, val_acc, _, _, _ = evaluate(
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
            save_model_checkpoint(
                model, optimizer, epoch, val_acc, args, 
                'best_spiking_resnet', args.save_dir, add_timestamp=False  # Folder already timestamped
            )
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    total_training_time = time.time() - training_start_time
    
    # Final evaluation with best model
    print("\nPerforming final comprehensive evaluation...")
    try:
        # Load the best model from the current run directory
        best_model_path = os.path.join(args.save_dir, 'best_spiking_resnet.pth')
        checkpoint = torch.load(best_model_path, weights_only=True)
    except Exception:
        print("Warning: Using unsafe loading due to checkpoint format compatibility")
        checkpoint = torch.load(best_model_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Comprehensive evaluation
    final_metrics = comprehensive_evaluate(model, val_loader, device)
    
    # Add training information
    final_metrics['training_time'] = total_training_time
    final_metrics['total_parameters'] = total_params
    final_metrics['model_size'] = model_size_mb
    final_metrics['best_val_acc'] = best_val_acc
    final_metrics['training_history'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    # Print comprehensive results
    print_comprehensive_results(final_metrics, "Spiking ResNet")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        final_metrics['y_true'], 
        final_metrics['y_pred'].flatten(),
        target_names=['Uncracked', 'Cracked']
    ))
    
    # Save final model
    save_model_checkpoint(
        model, optimizer, args.num_epochs-1, best_val_acc, args,
        'final_spiking_resnet', args.save_dir, add_timestamp=False  # Folder already timestamped
    )
    
    # Save evaluation results
    save_evaluation_results(
        final_metrics, 
        args.results_dir, 
        'spiking_resnet_training_results.json',
        add_timestamp=False  # Folder already timestamped
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Create a summary link in the base directory
    if run_name:
        create_run_summary(args, run_name, final_metrics)
    
    return final_metrics

def create_run_summary(args, run_name: str, metrics: dict):
    """Create a summary file in the base directory pointing to the timestamped run"""
    base_save_dir = os.path.dirname(args.save_dir)
    base_results_dir = os.path.dirname(args.results_dir)
    
    summary = {
        'run_name': run_name,
        'run_date': str(datetime.now()),
        'model_type': 'spiking_resnet',
        'best_accuracy': metrics.get('best_val_acc', 0),
        'total_training_time': metrics.get('training_time', 0),
        'directories': {
            'checkpoints': args.save_dir,
            'results': args.results_dir
        },
        'key_files': {
            'best_model': os.path.join(args.save_dir, 'best_spiking_resnet.pth'),
            'final_model': os.path.join(args.save_dir, 'final_spiking_resnet.pth'),
            'training_results': os.path.join(args.results_dir, 'spiking_resnet_training_results.json')
        },
        'configuration': vars(args)
    }
    
    # Save in base directories
    summary_path = os.path.join(base_save_dir, f'{run_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Run summary saved to: {summary_path}")

if __name__ == '__main__':
    main()