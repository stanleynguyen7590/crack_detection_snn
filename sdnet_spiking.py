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
        for t in range(self.T):
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

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Uncracked', 'Cracked'],
                yticklabels=['Uncracked', 'Cracked'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

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
        val_loss, val_acc, val_preds, val_targets = evaluate(
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
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
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
    plot_confusion_matrix(final_targets, final_preds)
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.save_dir, 'final_model.pth'))

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
    
    # Generate comparison table
    comparator.generate_crackvision_comparison_table()
    
    # Save results
    results_path = os.path.join(args.results_dir, 'baseline_comparison.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(comparator.results, f, indent=2, default=str)
    
    print(f"\nBaseline comparison results saved to {results_path}")
    return comparator.results

def run_comprehensive_evaluation_mode(args, device):
    """Run comprehensive evaluation combining CV and baseline comparison"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Run cross-validation
    print("\nPhase 1: Cross-Validation...")
    cv_results = run_cross_validation_mode(args, device)
    
    # Run baseline comparison
    print("\nPhase 2: Baseline Comparison...")
    baseline_results = run_baseline_comparison_mode(args, device)
    
    # Combined analysis
    print("\nPhase 3: Combined Analysis...")
    evaluator = ComprehensiveEvaluator()
    
    # Create comprehensive comparison table
    create_crackvision_style_table(baseline_results, "SDNET2018")
    
    # Save comprehensive results
    comprehensive_results = {
        'cross_validation': cv_results,
        'baseline_comparison': baseline_results,
        'evaluation_date': str(torch.datetime.now()),
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
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    start_time = time.time()
    
    # Simple training loop
    model.train()
    for epoch in range(min(args.num_epochs, 10)):  # Limit epochs for comparison
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
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
    
    return metrics

if __name__ == '__main__':
    main()
    # For inference on new images:
    # inference_example()