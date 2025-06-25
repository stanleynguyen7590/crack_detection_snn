# Manual Plotting Script Usage Examples

The `manual_plotting.py` script provides comprehensive visualization capabilities for analyzing saved models and results. Here are detailed usage examples:

## Prerequisites

Ensure you have the required dependencies and your models are trained and saved.

## Usage Modes

### 1. Single Model Analysis

Analyze a single saved model with comprehensive plots:

```bash
# Analyze a Spiking Neural Network model
python manual_plotting.py \
  --mode single \
  --model-path checkpoints/best_model.pth \
  --model-type snn \
  --model-name "Spiking_ResNet_18" \
  --data-dir ./SDNET2018 \
  --output-dir analysis_plots

# Analyze a CNN baseline model
python manual_plotting.py \
  --mode single \
  --model-path checkpoints/resnet50_baseline.pth \
  --model-type resnet50 \
  --model-name "ResNet50_Baseline" \
  --data-dir ./SDNET2018 \
  --output-dir cnn_analysis
```

**Generated plots for single model:**
- Confusion matrix
- ROC curve
- Class distribution comparison
- Prediction confidence analysis
- Calibration plot

### 2. Multiple Model Comparison

Compare multiple models using a configuration file:

```bash
# Create model configuration file (example_model_config.json)
python manual_plotting.py \
  --mode compare \
  --config-file example_model_config.json \
  --data-dir ./SDNET2018 \
  --output-dir comparison_plots \
  --batch-size 16
```

**Configuration file format (`example_model_config.json`):**
```json
[
  {
    "name": "Spiking ResNet-18",
    "path": "checkpoints/best_model.pth",
    "type": "snn"
  },
  {
    "name": "ResNet50 Baseline",
    "path": "checkpoints/resnet50_model.pth",
    "type": "resnet50"
  }
]
```

**Generated comparison plots:**
- Model performance comparison (accuracy, precision, recall, F1)
- ROC curves overlay
- Individual confusion matrices
- Performance summary table (CSV)

### 3. Plot from Saved Results

Generate plots from previously saved evaluation results:

```bash
# Plot from cross-validation results
python manual_plotting.py \
  --mode results \
  --results-file results/snn_cv_results.json \
  --output-dir cv_plots

# Plot from baseline comparison results
python manual_plotting.py \
  --mode results \
  --results-file results/baseline_comparison.json \
  --output-dir baseline_plots

# Plot from comprehensive evaluation results
python manual_plotting.py \
  --mode results \
  --results-file results/comprehensive_evaluation.json \
  --output-dir comprehensive_plots
```

### 4. Training History Plotting

Plot training curves from saved training history:

```bash
# Plot from JSON training history
python manual_plotting.py \
  --mode history \
  --history-file training_history.json \
  --model-name "SNN_Training" \
  --output-dir training_plots

# Plot from pickle file
python manual_plotting.py \
  --mode history \
  --history-file training_data.pkl \
  --model-name "CNN_Training" \
  --output-dir training_analysis
```

## Advanced Usage Examples

### Custom Analysis Pipeline

```bash
#!/bin/bash
# Comprehensive analysis script

# 1. Analyze best SNN model
python manual_plotting.py \
  --mode single \
  --model-path checkpoints/best_model.pth \
  --model-type snn \
  --model-name "Best_SNN" \
  --data-dir ./SDNET2018 \
  --output-dir final_analysis/snn

# 2. Compare all models
python manual_plotting.py \
  --mode compare \
  --config-file model_configs/all_models.json \
  --data-dir ./SDNET2018 \
  --output-dir final_analysis/comparison \
  --batch-size 32

# 3. Plot comprehensive evaluation results
python manual_plotting.py \
  --mode results \
  --results-file results/comprehensive_evaluation.json \
  --output-dir final_analysis/evaluation

echo "Analysis complete! Check final_analysis/ directory"
```

### GPU-Accelerated Analysis

```bash
# Use GPU for faster evaluation
python manual_plotting.py \
  --mode compare \
  --config-file large_model_config.json \
  --data-dir ./SDNET2018 \
  --device cuda \
  --batch-size 64 \
  --num-workers 8 \
  --output-dir gpu_analysis
```

## Output Directory Structure

The script organizes outputs in a clear directory structure:

```
manual_plots/
├── model_comparison.png                    # Overall metrics comparison
├── roc_curves_comparison.png              # ROC curves overlay  
├── confusion_matrix_[model_name].png      # Per-model confusion matrices
├── performance_comparison.csv             # Summary table
├── spiking_resnet_18/                     # Single model analysis
│   ├── confusion_matrix_spiking_resnet_18.png
│   ├── roc_curve_spiking_resnet_18.png
│   ├── class_distribution_spiking_resnet_18.png
│   └── confidence_analysis_spiking_resnet_18.png
└── resnet50_baseline/
    ├── confusion_matrix_resnet50_baseline.png
    ├── roc_curve_resnet50_baseline.png
    ├── class_distribution_resnet50_baseline.png
    └── confidence_analysis_resnet50_baseline.png
```

## Plot Types and Features

### 1. Confusion Matrix
- Normalized and absolute counts
- Class-specific accuracy visualization
- High-resolution heatmap with annotations

### 2. ROC Curves
- Individual and comparison ROC curves
- AUC scores displayed
- Perfect classifier reference line

### 3. Model Comparison Charts
- 4-panel comparison: Accuracy, Precision, Recall, F1-Score
- Bar charts with value annotations
- Color-coded for easy identification

### 4. Confidence Analysis
- Overall confidence distribution
- Confidence by prediction correctness
- Confidence by true class
- Model calibration plot (reliability diagram)

### 5. Class Distribution
- True vs predicted class distributions
- Helps identify class imbalance effects
- Visual comparison of prediction bias

### 6. Training History
- Loss and accuracy curves
- Train vs validation comparison
- Epoch-wise progression analysis

## Customization Options

### Model Types Supported
- `snn`: Spiking Neural Networks
- `resnet50`: ResNet-50 CNN
- `resnet18`: ResNet-18 CNN  
- `xception`: Xception-style CNN
- `cnn`: Generic CNN (defaults to ResNet-50)

### Device Options
- `auto`: Automatically detect GPU/CPU
- `cpu`: Force CPU usage
- `cuda`: Force GPU usage

### Data Loading Options
- `--batch-size`: Adjust batch size for memory constraints
- `--num-workers`: Parallel data loading workers
- `--train-ratio`: Train/validation split ratio

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Ensure correct model type
   python manual_plotting.py --mode single --model-path model.pth --model-type snn
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python manual_plotting.py --mode single --batch-size 8 --device cpu
   ```

3. **Missing Dependencies**
   ```bash
   pip install matplotlib seaborn scikit-learn pandas
   ```

### Performance Tips

1. **Use GPU for Large Models**
   ```bash
   --device cuda --batch-size 64
   ```

2. **Parallel Data Loading**
   ```bash
   --num-workers 8
   ```

3. **Efficient Evaluation**
   ```bash
   --batch-size 32  # Balance speed vs memory
   ```

## Integration with Main Training

The manual plotting script integrates seamlessly with the main training pipeline:

```python
# In your training script
import json

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses, 
    'train_accs': train_accs,
    'val_accs': val_accs
}

with open('training_history.json', 'w') as f:
    json.dump(history, f)

# Later, plot the history
# python manual_plotting.py --mode history --history-file training_history.json
```

This comprehensive plotting tool enables detailed post-hoc analysis of your models and results, supporting both research and presentation needs.