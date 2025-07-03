#!/usr/bin/env python3
"""
CNN Baseline Training for SDNET2018 Concrete Crack Detection
Following CrackVision methodology

Focused on training and evaluation of CNN baseline models.
Supports ResNet50, ResNet18, Xception-style, and InceptionV3 architectures.

Dataset: SDNET2018 - 56,000+ annotated concrete crack images (256x256)
Framework: PyTorch with torchvision models
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Import evaluation modules
from evaluation_metrics import ComprehensiveEvaluator
from cross_validation import CrossValidator, run_cross_validation_experiment

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

class CNNBaseline(nn.Module):
    """CNN baseline models (ResNet50, ResNet18, Xception-style, InceptionV3)"""
    
    def __init__(self, architecture: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        elif architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        elif architecture == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
        elif architecture == 'xception_style':
            # Simplified Xception-style architecture
            self.backbone = self._create_xception_style(num_classes)
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def _create_xception_style(self, num_classes: int) -> nn.Module:
        """Create a simplified Xception-style model"""
        return nn.Sequential(
            # Entry flow
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions
            self._depthwise_separable_conv(64, 128, stride=2),
            self._depthwise_separable_conv(128, 256, stride=2),
            self._depthwise_separable_conv(256, 512, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def _depthwise_separable_conv(self, in_channels: int, out_channels: int, 
                                 stride: int = 1) -> nn.Module:
        """Depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, 
                       train_ratio: float = 0.8, input_size: int = 224):
    """Create data loaders with appropriate transforms for CNN models"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # Slightly larger than crop
        transforms.CenterCrop(input_size),
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
        'model_type': 'cnn',
        'architecture': getattr(model, 'architecture', 'unknown'),
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

def print_comprehensive_results(metrics, model_name="CNN Baseline"):
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

def get_input_size(architecture: str) -> int:
    """Get appropriate input size for different architectures"""
    if architecture == 'inception_v3':
        return 299  # InceptionV3 expects 299x299
    else:
        return 224  # Standard ImageNet size for ResNet, Xception-style

def create_model(architecture: str, num_classes: int = 2, pretrained: bool = True) -> CNNBaseline:
    """Create CNN model with specified architecture"""
    return CNNBaseline(architecture=architecture, num_classes=num_classes, pretrained=pretrained)

def train_single_model(args, device, architecture: str, use_timestamps: bool = True):
    """Train a single CNN model"""
    print(f"\n{'='*60}")
    print(f"TRAINING {architecture.upper()} MODEL")
    print(f"{'='*60}")
    
    # Get appropriate input size
    input_size = get_input_size(architecture)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        input_size=input_size
    )
    
    # Create model
    model = create_model(architecture, args.num_classes, args.pretrained).to(device)
    
    # Model size information
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Create model name for saving
    model_name = f'{architecture}_baseline'
    
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
        metrics['architecture'] = architecture
        
        # Print and save results
        print_comprehensive_results(metrics, f"{architecture} (Pre-trained)")
        save_evaluation_results(
            metrics, 
            args.results_dir, 
            f'{architecture}_eval_only.json', 
            add_timestamp=False  # Folder already timestamped
        )
        
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
    
    # Different learning rates for different architectures
    if architecture == 'inception_v3':
        lr = args.learning_rate * 0.1  # Lower learning rate for Inception
    else:
        lr = args.learning_rate
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
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
    print(f"Learning rate: {lr}")
    
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
                f'best_{model_name}', args.save_dir, add_timestamp=False  # Folder already timestamped
            )
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    total_training_time = time.time() - training_start_time
    
    # Final evaluation with best model
    print("\nPerforming final comprehensive evaluation...")
    try:
        # Load the best model from the current run directory
        best_model_path = os.path.join(args.save_dir, f'best_{model_name}.pth')
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
    final_metrics['architecture'] = architecture
    final_metrics['training_history'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    # Print comprehensive results
    print_comprehensive_results(final_metrics, f"{architecture}")
    
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
        f'final_{model_name}', args.save_dir, add_timestamp=False  # Folder already timestamped
    )
    
    # Save evaluation results
    save_evaluation_results(
        final_metrics, 
        args.results_dir, 
        f'{architecture}_training_results.json',
        add_timestamp=False  # Folder already timestamped
    )
    
    print(f"\n{architecture} training completed successfully!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return final_metrics

def train_all_models(args, device, use_timestamps: bool = True):
    """Train all CNN baseline models"""
    all_results = {}
    
    architectures = ['resnet18', 'resnet50', 'xception_style']
    if not args.skip_inception:
        architectures.append('inception_v3')
    
    for architecture in architectures:
        print(f"\n{'#'*80}")
        print(f"# TRAINING {architecture.upper()}")
        print(f"{'#'*80}")
        
        try:
            results = train_single_model(args, device, architecture, use_timestamps)
            all_results[architecture] = results
            
            # Save intermediate results
            save_evaluation_results(
                all_results,
                args.results_dir,
                'baseline_models_intermediate.json',
                add_timestamp=False  # Folder already timestamped
            )
            
        except Exception as e:
            print(f"Error training {architecture}: {e}")
            continue
    
    # Save final comprehensive results
    final_results = {
        'all_models': all_results,
        'evaluation_date': str(datetime.now()),
        'configuration': vars(args)
    }
    
    save_evaluation_results(
        final_results,
        args.results_dir,
        'baseline_models_comprehensive.json',
        add_timestamp=False  # Folder already timestamped
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("BASELINE MODELS TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for arch, results in all_results.items():
        if results:
            acc = results.get('accuracy', 0) * 100 if results.get('accuracy', 0) <= 1 else results.get('accuracy', 0)
            time_sec = results.get('training_time', 0)
            print(f"{arch:15s}: {acc:6.2f}% accuracy, {time_sec:8.1f}s training time")
    
    return final_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CNN Baseline Training for SDNET2018 Concrete Crack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, 
                       default='/home/duyanh/Workspace/SDNET_spiking/SDNET2018',
                       help='Path to SDNET2018 dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of output classes')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable pretrained weights')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='all',
                       choices=['all', 'resnet18', 'resnet50', 'xception_style', 'inception_v3'],
                       help='CNN architecture to train')
    parser.add_argument('--skip-inception', action='store_true',
                       help='Skip InceptionV3 training (requires different input size)')
    
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
        run_name = f"cnn_run_{timestamp}"
        
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

def create_run_summary(args, run_name: str, results: dict, architecture: str = None):
    """Create a summary file in the base directory pointing to the timestamped run"""
    base_save_dir = os.path.dirname(args.save_dir)
    base_results_dir = os.path.dirname(args.results_dir)
    
    # Handle both single model and multi-model results
    if isinstance(results, dict) and 'all_models' in results:
        # Multi-model results
        best_model = None
        best_acc = 0
        for arch, metrics in results['all_models'].items():
            if metrics and metrics.get('best_val_acc', 0) > best_acc:
                best_acc = metrics.get('best_val_acc', 0)
                best_model = arch
        
        summary = {
            'run_name': run_name,
            'run_date': str(datetime.now()),
            'model_type': 'cnn_baselines_all',
            'architectures_trained': list(results['all_models'].keys()),
            'best_model': best_model,
            'best_accuracy': best_acc,
            'directories': {
                'checkpoints': args.save_dir,
                'results': args.results_dir
            },
            'configuration': vars(args)
        }
    else:
        # Single model results
        summary = {
            'run_name': run_name,
            'run_date': str(datetime.now()),
            'model_type': f'cnn_{architecture}' if architecture else 'cnn_baseline',
            'architecture': architecture,
            'best_accuracy': results.get('best_val_acc', 0) if results else 0,
            'total_training_time': results.get('training_time', 0) if results else 0,
            'directories': {
                'checkpoints': args.save_dir,
                'results': args.results_dir
            },
            'configuration': vars(args)
        }
    
    # Save in base directories
    summary_path = os.path.join(base_save_dir, f'{run_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Run summary saved to: {summary_path}")

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
    
    # Set pretrained flag
    args.pretrained = not args.no_pretrained
    print(f"Using pretrained weights: {args.pretrained}")
    
    # Create timestamped directories
    save_dir, results_dir, run_name = create_timestamped_directories(
        args.save_dir, args.results_dir, use_timestamps
    )
    
    # Update args with new directories
    args.save_dir = save_dir
    args.results_dir = results_dir
    
    # Train models based on architecture choice
    if args.architecture == 'all':
        results = train_all_models(args, device, use_timestamps)
        if run_name:
            create_run_summary(args, run_name, results)
        return results
    else:
        results = train_single_model(args, device, args.architecture, use_timestamps)
        if run_name:
            create_run_summary(args, run_name, results, args.architecture)
        return results

if __name__ == '__main__':
    main()