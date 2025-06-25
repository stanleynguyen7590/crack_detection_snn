# SDNET Spiking Neural Network - User Guide

## Overview

This project implements a Spiking Neural Network (SNN) for concrete crack detection using the SDNET2018 dataset. The implementation uses SpikingJelly framework with PyTorch backend and features a Spiking ResNet-18 architecture for binary classification (cracked vs uncracked concrete).

### Key Features

- **Spiking ResNet-18 Architecture**: Bio-inspired temporal dynamics processing
- **SDNET2018 Dataset Support**: 56,000+ annotated concrete crack images (256x256)
- **Comprehensive Evaluation Framework**: Cross-validation, baseline comparisons, and statistical analysis
- **CrackVision-Inspired Methodology**: Publication-ready evaluation following academic standards
- **Multiple Baseline Models**: CNN comparisons (ResNet50, ResNet18, Xception-style)
- **Interactive Experiment Runner**: Easy-to-use evaluation interface

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Required Dependencies

```bash
pip install torch torchvision torchaudio
pip install spikingjelly
pip install scikit-learn
pip install matplotlib seaborn
pip install opencv-python
pip install pillow
pip install pandas
pip install scipy
```

### Dataset Setup

1. Download the SDNET2018 dataset
2. Extract to your desired location
3. Ensure the directory structure follows:
```
SDNET2018/
├── D/ (Decks)
│   ├── CD/ (Cracked Decks)
│   └── UD/ (Uncracked Decks)
├── P/ (Pavements)
│   ├── CP/ (Cracked Pavements)
│   └── UP/ (Uncracked Pavements)
└── W/ (Walls)
    ├── CW/ (Cracked Walls)
    └── UW/ (Uncracked Walls)
```

## Quick Start

### Basic Training

Run with default parameters:
```bash
python sdnet_spiking.py
```

### Custom Training Parameters

```bash
python sdnet_spiking.py \
  --data-dir /path/to/SDNET2018 \
  --batch-size 16 \
  --learning-rate 0.0005 \
  --num-epochs 50 \
  --time-steps 8
```

### Evaluation Modes

#### 1. Cross-Validation Evaluation
```bash
python sdnet_spiking.py --eval-mode cross_validation --cv-folds 5
```

#### 2. Baseline Comparison
```bash
python sdnet_spiking.py --eval-mode baseline_comparison
```

#### 3. Comprehensive Evaluation
```bash
python sdnet_spiking.py --eval-mode comprehensive
```

#### 4. Interactive Experiment Runner
```bash
python run_evaluation.py
```

For quick testing:
```bash
python run_evaluation.py --quick-test
```

## Command Line Arguments

### Data Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | ./SDNET2018 | Path to SDNET2018 dataset |
| `--batch-size` | int | 8 | Training batch size |
| `--num-workers` | int | 4 | Data loading workers |
| `--train-ratio` | float | 0.8 | Training/validation split ratio |

### Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-epochs` | int | 20 | Number of training epochs |
| `--learning-rate`, `--lr` | float | 1e-3 | Learning rate |
| `--weight-decay` | float | 1e-4 | L2 regularization |

### Model Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--time-steps`, `-T` | int | 10 | Spiking network time steps |
| `--num-classes` | int | 2 | Number of output classes |

### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | auto | Training device (auto/cpu/cuda) |
| `--no-amp` | flag | False | Disable mixed precision training |
| `--save-dir` | str | checkpoints | Checkpoint save directory |
| `--resume` | str | None | Resume from checkpoint path |

### Evaluation Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval-mode` | str | train | Evaluation mode (train/cross_validation/baseline_comparison/comprehensive) |
| `--cv-folds` | int | 5 | Number of cross-validation folds |
| `--results-dir` | str | results | Results save directory |

## Usage Examples

### Example 1: Standard Training
```bash
python sdnet_spiking.py \
  --data-dir ./SDNET2018 \
  --batch-size 16 \
  --num-epochs 30 \
  --learning-rate 0.001 \
  --time-steps 10
```

### Example 2: Cross-Validation with Custom Parameters
```bash
python sdnet_spiking.py \
  --eval-mode cross_validation \
  --cv-folds 10 \
  --batch-size 12 \
  --time-steps 8 \
  --results-dir ./my_results
```

### Example 3: CPU-Only Training
```bash
python sdnet_spiking.py \
  --device cpu \
  --no-amp \
  --batch-size 4 \
  --num-epochs 10
```

### Example 4: Resume Training
```bash
python sdnet_spiking.py \
  --resume checkpoints/best_model.pth \
  --num-epochs 50
```

### Example 5: Comprehensive Evaluation
```bash
python sdnet_spiking.py \
  --eval-mode comprehensive \
  --data-dir /path/to/SDNET2018 \
  --results-dir ./final_results
```

## Architecture Details

### Spiking ResNet-18 Components

1. **Temporal Processing**: Input images repeated across T timesteps
2. **Spiking Neurons**: LIF (Leaky Integrate-and-Fire) neurons with surrogate gradients
3. **Residual Connections**: Adapted for temporal dynamics
4. **Output Integration**: Temporal averaging of spike trains
5. **State Reset**: Network state reset after each forward pass

### Key Classes

- **`SDNET2018Dataset`**: Custom PyTorch Dataset for hierarchical data loading
- **`SpikingResNetCrackDetector`**: Main SNN model using SpikingJelly ResNet-18
- **`DirectEncodingSpikingResNet`**: Alternative architecture with direct encoding
- **`ComprehensiveEvaluator`**: Advanced metrics calculation
- **`CrossValidator`**: 5-fold stratified cross-validation
- **`CNNBaseline`**: CNN baseline models for comparison

## Output Files

### Training Outputs
- `checkpoints/best_model.pth` - Best model checkpoint
- `checkpoints/final_model.pth` - Final epoch checkpoint
- `training_history.png` - Loss and accuracy plots
- `confusion_matrix.png` - Confusion matrix visualization

### Evaluation Outputs
- `results/snn_cv_results.json` - Cross-validation results
- `results/baseline_comparison.json` - Model comparison
- `results/comprehensive_evaluation.json` - Combined analysis
- `results/plots/` - ROC curves, performance charts

## Performance Optimization

### Memory Optimization
- Use smaller batch sizes for limited GPU memory
- Enable mixed precision training (default)
- Reduce time steps if memory constrained

### Speed Optimization
- Increase batch size for faster training
- Use multiple data loading workers
- Enable CUDA if available

### Recommended Settings

**For RTX 3080/4080 (12GB VRAM):**
```bash
--batch-size 16 --time-steps 10 --num-workers 8
```

**For RTX 3060 (8GB VRAM):**
```bash
--batch-size 8 --time-steps 8 --num-workers 4
```

**For CPU-only systems:**
```bash
--device cpu --no-amp --batch-size 4 --num-workers 2
```

## Evaluation Metrics

The framework calculates comprehensive metrics including:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average precision
- **Recall**: Per-class and weighted average recall
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification breakdown
- **Statistical Tests**: Cross-validation significance testing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 4`
   - Reduce time steps: `--time-steps 6`
   - Disable mixed precision: `--no-amp`

2. **Dataset Not Found**
   - Check dataset path: `--data-dir /correct/path/to/SDNET2018`
   - Verify directory structure matches expected format

3. **Slow Training**
   - Increase batch size if memory allows
   - Use GPU: ensure CUDA is properly installed
   - Increase data loading workers: `--num-workers 8`

4. **Checkpoint Loading Issues**
   - Ensure checkpoint path is correct
   - Check model architecture compatibility

### Performance Tips

- **Data Loading**: Use SSD storage for faster data loading
- **Batch Size**: Experiment with batch sizes (4, 8, 16, 32)
- **Time Steps**: Balance between accuracy and computational cost
- **Mixed Precision**: Keep enabled unless encountering numerical issues

## Advanced Usage

### Model Inference
```python
from sdnet_spiking import inference_example
inference_example()  # Load trained model and predict on sample images
```

### Custom Dataset Integration
Modify the `SDNET2018Dataset` class to work with your own crack detection dataset:
```python
class CustomCrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Implement custom data loading logic
        pass
```

### Hyperparameter Tuning
Use the interactive runner for systematic hyperparameter exploration:
```bash
python run_evaluation.py
# Follow prompts to test different configurations
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sdnet_spiking_2024,
  title={Spiking Neural Networks for Concrete Crack Detection},
  author={Your Name},
  year={2024},
  note={Implementation based on SDNET2018 dataset}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the command line help: `python sdnet_spiking.py --help`
3. Examine the evaluation results in the `results/` directory
4. Open an issue on the project repository

## Acknowledgments

- SDNET2018 dataset creators
- SpikingJelly framework developers
- CrackVision paper for evaluation methodology
- PyTorch and scikit-learn communities