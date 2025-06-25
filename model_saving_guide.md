# Model Saving and Loading Guide

## Overview

The training script now saves models with consistent naming and metadata for easy loading and analysis.

## Model File Formats

### 1. SNN Models (from main training)

**Files:**
- `checkpoints/best_model.pth` - Best validation accuracy model
- `checkpoints/final_model.pth` - Final epoch model

**Format:**
```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': dict,        # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state (best_model only)
    'val_acc': float,               # Validation accuracy
    'args': Namespace,              # Training arguments
    'model_type': 'snn'             # Model type identifier
}
```

### 2. CNN Baseline Models (from baseline comparison)

**Files:**
- `checkpoints/resnet50_crackvision_baseline.pth`
- `checkpoints/resnet18_baseline.pth`
- `checkpoints/xception_style_baseline.pth`
- `checkpoints/spiking_resnet_baseline.pth` (from baseline comparison)

**Format:**
```python
{
    'model_state_dict': dict,        # Model parameters
    'model_name': str,              # Human-readable name
    'model_type': str,              # Architecture type
    'val_acc': float,               # Validation accuracy
    'metrics': dict                 # Additional metrics
}
```

## Loading Models

### Using Manual Plotting Script

The manual plotting script automatically handles both formats:

```bash
# Load SNN model
python manual_plotting.py \
  --mode single \
  --model-path checkpoints/best_model.pth \
  --model-type snn \
  --model-name "Best_SNN"

# Load CNN baseline
python manual_plotting.py \
  --mode single \
  --model-path checkpoints/resnet50_crackvision_baseline.pth \
  --model-type resnet50 \
  --model-name "ResNet50_Baseline"
```

### Manual Loading in Python

```python
import torch
from sdnet_spiking import SpikingResNetCrackDetector
from baseline_models import CNNBaseline

# Load SNN model
def load_snn_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract parameters from checkpoint
    args = checkpoint.get('args')
    time_steps = args.time_steps if args else 10
    
    model = SpikingResNetCrackDetector(
        num_classes=2,
        T=time_steps
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded SNN model with {checkpoint['val_acc']:.2f}% accuracy")
    return model

# Load CNN baseline
def load_cnn_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_type = checkpoint.get('model_type', 'resnet50')
    model = CNNBaseline(model_type, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded {checkpoint['model_name']} with {checkpoint['val_acc']:.2f}% accuracy")
    return model

# Usage examples
snn_model = load_snn_model('checkpoints/best_model.pth')
cnn_model = load_cnn_model('checkpoints/resnet50_crackvision_baseline.pth')
```

## Model Discovery

### Automatic Discovery

The quick plotting script automatically finds all models:

```bash
python quick_plot.py
# Select option 1 or 5 for automatic model discovery
```

### Manual Discovery

```python
import os
from pathlib import Path

def find_all_models(directory="checkpoints"):
    """Find all model files"""
    models = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                models.append(os.path.join(root, file))
    return models

# Find all models
all_models = find_all_models()
for model in all_models:
    print(f"Found: {model}")
```

## Model Comparison Configuration

### Updated example_model_config.json

```json
[
  {
    "name": "Spiking ResNet-18",
    "path": "checkpoints/best_model.pth",
    "type": "snn"
  },
  {
    "name": "ResNet50 Baseline",
    "path": "checkpoints/resnet50_crackvision_baseline.pth",
    "type": "resnet50"
  },
  {
    "name": "ResNet18 Baseline", 
    "path": "checkpoints/resnet18_baseline.pth",
    "type": "resnet18"
  },
  {
    "name": "Xception-style CNN",
    "path": "checkpoints/xception_style_baseline.pth", 
    "type": "xception"
  }
]
```

## Training Commands and Outputs

### Standard Training
```bash
python sdnet_spiking.py --num-epochs 20 --batch-size 16
```
**Saves:**
- `checkpoints/best_model.pth`
- `checkpoints/final_model.pth`
- `checkpoints/plots/training_history.png`
- `checkpoints/plots/confusion_matrix_spiking_resnet.png`

### Baseline Comparison
```bash
python sdnet_spiking.py --eval-mode baseline_comparison
```
**Saves:**
- `checkpoints/spiking_resnet_baseline.pth`
- `checkpoints/resnet50_crackvision_baseline.pth`
- `checkpoints/resnet18_baseline.pth`
- `checkpoints/xception_style_baseline.pth`
- `results/baseline_comparison.json`
- `results/plots/model_comparison.png`
- `results/plots/efficiency_comparison.png`

### Comprehensive Evaluation
```bash
python sdnet_spiking.py --eval-mode comprehensive --cv-folds 5
```
**Saves:**
- All models from baseline comparison
- `results/snn_cv_results.json`
- `results/comprehensive_evaluation.json`
- `results/plots/cross_validation_results.png`
- All comparison plots

## Model Metadata

Each saved model includes metadata for easy identification:

### SNN Models
- Training epoch and validation accuracy
- Original training arguments
- Model architecture parameters
- Model type identifier

### CNN Baselines
- Model name and architecture
- Final validation accuracy
- Comprehensive evaluation metrics
- Training time and model size

## Best Practices

1. **Always use the plotting scripts** - They handle all model formats automatically
2. **Check model metadata** - Validation accuracy and training info are preserved
3. **Use consistent naming** - The system automatically generates safe filenames
4. **Organize by experiment** - Use different `--save-dir` for different experiments
5. **Backup important models** - Copy best performing models to a separate directory

## Troubleshooting

### Common Issues

1. **"KeyError: model_state_dict"**
   - Old model format, update training script
   - Solution: Use updated loading code that handles both formats

2. **Model architecture mismatch**
   - Ensure correct model type in config
   - Check time_steps parameter for SNN models

3. **Missing baseline models**
   - Run baseline comparison mode first
   - Check the `checkpoints/` directory

### Migration from Old Format

If you have models saved in the old format:

```python
# Convert old format to new format
old_model = torch.load('old_model.pth')
new_model = {
    'model_state_dict': old_model,  # old_model was just state_dict
    'model_type': 'snn',
    'val_acc': 0.0  # Unknown accuracy
}
torch.save(new_model, 'new_model.pth')
```

This improved model saving system ensures consistency, traceability, and easy integration with the plotting tools.