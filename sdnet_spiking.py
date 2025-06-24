"""
Spiking ResNet Implementation for SDNET2018 Concrete Crack Detection
Using SpikingJelly Framework

Dataset: SDNET2018 - 56,000+ annotated concrete crack images (256x256)
Framework: SpikingJelly with PyTorch backend
Architecture: Spiking ResNet-18 adapted for binary classification
"""

import os
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

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
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
    train_dataset = SDNET2018Dataset(data_dir, split='train', transform=train_transform)
    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform)
    
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
            with torch.cuda.amp.autocast():
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

def main():
    """Main training function"""
    # Configuration
    config = {
        'data_dir': '/home/duyanh/Workspace/SDNET_spiking/SDNET2018',  # Update this path
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'T': 10,  # Time steps
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': True,  # Automatic mixed precision
        'save_dir': 'checkpoints'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        config['data_dir'], 
        batch_size=config['batch_size']
    )
    
    # Create model
    model = SpikingResNetCrackDetector(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=2,
        T=config['T']
    ).to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            config['device'], scaler
        )
        
        # Evaluate
        val_loss, val_acc, val_preds, val_targets = evaluate(
            model, val_loader, criterion, config['device']
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
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Final evaluation with best model
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, final_preds, final_targets = evaluate(
        model, val_loader, criterion, config['device']
    )
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_targets, final_preds, 
                              target_names=['Uncracked', 'Cracked']))
    
    # Confusion matrix
    plot_confusion_matrix(final_targets, final_preds)
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(config['save_dir'], 'final_model.pth'))

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

if __name__ == '__main__':
    main()
    # For inference on new images:
    # inference_example()