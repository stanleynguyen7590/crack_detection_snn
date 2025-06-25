"""
Baseline Models for Comparison with SNN
Implements CNN baselines following CrackVision methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import cv2
from typing import Dict, Any, Tuple
import time

class CNNBaseline(nn.Module):
    """CNN baseline models (ResNet50, Xception-style, InceptionV3)"""
    
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

class TraditionalBaseline:
    """Traditional computer vision methods for crack detection"""
    
    def __init__(self, method: str = 'canny_svm', **kwargs):
        self.method = method
        self.classifier = None
        self.feature_extractor = None
        
        if method == 'canny_svm':
            self.classifier = SVC(probability=True, **kwargs)
            self.feature_extractor = self._canny_features
            
        elif method == 'hog_svm':
            self.classifier = SVC(probability=True, **kwargs)
            self.feature_extractor = self._hog_features
            
        elif method == 'lbp_rf':
            self.classifier = RandomForestClassifier(**kwargs)
            self.feature_extractor = self._lbp_features
            
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _canny_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Canny edge features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate feature statistics
        features = [
            np.mean(edges),  # Mean edge intensity
            np.std(edges),   # Standard deviation
            np.sum(edges > 0) / edges.size,  # Edge density
            np.max(edges),   # Maximum edge intensity
            np.sum(edges == 255) / edges.size,  # Strong edge density
        ]
        
        # Add spatial statistics
        edge_rows = np.sum(edges, axis=1)
        edge_cols = np.sum(edges, axis=0)
        
        features.extend([
            np.std(edge_rows),  # Row variation
            np.std(edge_cols),  # Column variation
            np.max(edge_rows),  # Maximum row edges
            np.max(edge_cols),  # Maximum column edges
        ])
        
        return np.array(features)
    
    def _hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to fixed size for consistency
        gray = cv2.resize(gray, (128, 128))
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor(_winSize=(128, 128),
                               _blockSize=(16, 16),
                               _blockStride=(8, 8),
                               _cellSize=(8, 8),
                               _nbins=9)
        
        features = hog.compute(gray)
        
        return features.flatten()
    
    def _lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                
                # 8 neighbors
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def train(self, images: list, labels: list) -> Dict[str, Any]:
        """
        Train the traditional baseline model
        
        Args:
            images: List of images (numpy arrays)
            labels: List of corresponding labels
            
        Returns:
            Training statistics
        """
        print(f"Extracting features using {self.method}...")
        start_time = time.time()
        
        features = []
        for i, image in enumerate(images):
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(images)}")
            
            feature = self.feature_extractor(image)
            features.append(feature)
        
        features = np.array(features)
        labels = np.array(labels)
        
        feature_time = time.time() - start_time
        print(f"Feature extraction completed in {feature_time:.2f} seconds")
        
        # Train classifier
        print("Training classifier...")
        start_time = time.time()
        self.classifier.fit(features, labels)
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.2f} seconds")
        
        return {
            'feature_extraction_time': feature_time,
            'training_time': train_time,
            'feature_dimension': features.shape[1],
            'num_samples': len(labels)
        }
    
    def predict(self, images: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test images
        
        Args:
            images: List of test images
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        features = []
        for image in images:
            feature = self.feature_extractor(image)
            features.append(feature)
        
        features = np.array(features)
        
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        
        return predictions, probabilities
    
    def evaluate(self, images: list, labels: list) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            images: List of test images
            labels: List of true labels
            
        Returns:
            Evaluation metrics
        """
        start_time = time.time()
        predictions, probabilities = self.predict(images)
        inference_time = time.time() - start_time
        
        from evaluation_metrics import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        
        metrics = evaluator.calculate_comprehensive_metrics(
            np.array(labels), predictions, probabilities
        )
        
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_image'] = inference_time / len(images)
        
        return metrics

class ModelComparator:
    """Compare different models following CrackVision methodology"""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """Add results for a model"""
        self.results[model_name] = results
    
    def compare_computational_efficiency(self) -> Dict[str, Any]:
        """Compare computational efficiency across models"""
        comparison = {
            'training_time': {},
            'inference_time': {},
            'memory_usage': {},
            'model_size': {}
        }
        
        for model_name, results in self.results.items():
            if 'training_time' in results:
                comparison['training_time'][model_name] = results['training_time']
            if 'inference_time' in results:
                comparison['inference_time'][model_name] = results['inference_time']
            if 'memory_usage' in results:
                comparison['memory_usage'][model_name] = results['memory_usage']
            if 'model_size' in results:
                comparison['model_size'][model_name] = results['model_size']
        
        return comparison
    
    def generate_crackvision_comparison_table(self) -> None:
        """Generate a comparison table similar to CrackVision Table 18"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON (CrackVision Style)")
        print("="*80)
        
        # Performance metrics
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
        print("-"*80)
        
        for model_name, results in self.results.items():
            acc = f"{results.get('accuracy', 0)*100:.2f}%"
            prec = f"{results.get('precision', 0)*100:.2f}%"
            rec = f"{results.get('recall', 0)*100:.2f}%"
            f1 = f"{results.get('f1_score', 0)*100:.2f}%"
            auc = f"{results.get('auc_roc', 0):.3f}" if results.get('auc_roc') else 'N/A'
            
            print(f"{model_name:<20} {acc:<10} {prec:<10} {rec:<10} {f1:<10} {auc:<10}")
        
        print("\n" + "="*80)
        print("COMPUTATIONAL EFFICIENCY COMPARISON")
        print("="*80)
        
        # Efficiency metrics
        print(f"{'Model':<20} {'Train Time':<12} {'Inference':<12} {'Memory':<10} {'Size':<10}")
        print("-"*80)
        
        for model_name, results in self.results.items():
            train_time = f"{results.get('training_time', 0):.1f}s" if 'training_time' in results else 'N/A'
            inf_time = f"{results.get('inference_time_per_image', 0)*1000:.1f}ms" if 'inference_time_per_image' in results else 'N/A'
            memory = f"{results.get('memory_usage', 0):.1f}MB" if 'memory_usage' in results else 'N/A'
            size = f"{results.get('model_size', 0):.1f}MB" if 'model_size' in results else 'N/A'
            
            print(f"{model_name:<20} {train_time:<12} {inf_time:<12} {memory:<10} {size:<10}")

def create_crackvision_baseline():
    """Create ResNet50 baseline matching CrackVision's best performer"""
    return CNNBaseline(architecture='resnet50', num_classes=2, pretrained=True)

def create_traditional_baselines():
    """Create traditional method baselines"""
    return {
        'Canny + SVM': TraditionalBaseline('canny_svm'),
        'HOG + SVM': TraditionalBaseline('hog_svm'),
        'LBP + RandomForest': TraditionalBaseline('lbp_rf')
    }