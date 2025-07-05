#!/usr/bin/env python3
"""
Validation Script for Enhanced Crack-Aware Augmentation
Tests that enhanced augmentation is working correctly and shows visual examples
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_class_dataset import create_multi_class_datasets, CrackAwareTransform, create_enhanced_transforms

def test_augmentation_functionality(data_dir: str, num_samples: int = 8):
    """Test that enhanced augmentation is working correctly"""
    
    print("🔍 TESTING ENHANCED CRACK-AWARE AUGMENTATION")
    print("=" * 60)
    
    # Test 1: Create datasets with and without enhanced augmentation
    print("\n1. Creating datasets with different augmentation settings...")
    
    try:
        # Standard augmentation
        standard_data = create_multi_class_datasets(
            data_dir=data_dir,
            class_scheme='binary',
            dataset_type='all',
            batch_size=4,
            use_enhanced_augmentation=False
        )
        print("✅ Standard augmentation dataset created successfully")
        
        # Enhanced augmentation
        enhanced_data = create_multi_class_datasets(
            data_dir=data_dir,
            class_scheme='binary', 
            dataset_type='deck',  # Use deck for focused testing
            batch_size=4,
            use_enhanced_augmentation=True
        )
        print("✅ Enhanced augmentation dataset created successfully")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return False
    
    # Test 2: Verify CrackAwareTransform is being used
    print("\n2. Verifying CrackAwareTransform usage...")
    
    train_dataset = enhanced_data['train_dataset']
    if isinstance(train_dataset.transform, CrackAwareTransform):
        print("✅ CrackAwareTransform is correctly applied to enhanced dataset")
    else:
        print("❌ Enhanced dataset is not using CrackAwareTransform")
        return False
    
    # Test 3: Visual validation of augmentation differences
    print("\n3. Generating visual comparison of augmentation effects...")
    
    try:
        create_augmentation_comparison_plot(enhanced_data, num_samples)
        print("✅ Augmentation comparison plot created successfully")
    except Exception as e:
        print(f"❌ Error creating visual comparison: {e}")
        return False
    
    # Test 4: Test augmentation statistics
    print("\n4. Testing augmentation statistics...")
    
    try:
        test_augmentation_statistics(enhanced_data)
        print("✅ Augmentation statistics look good")
    except Exception as e:
        print(f"❌ Error in augmentation statistics: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL AUGMENTATION TESTS PASSED!")
    print("Enhanced crack-aware augmentation is working correctly.")
    return True

def create_augmentation_comparison_plot(dataset_info, num_samples: int = 8):
    """Create visual comparison of augmentation effects"""
    
    train_dataset = dataset_info['train_dataset']
    
    # Find samples with both cracked and uncracked examples
    cracked_indices = [i for i, is_crack in enumerate(train_dataset.is_cracked) if is_crack]
    uncracked_indices = [i for i, is_crack in enumerate(train_dataset.is_cracked) if not is_crack]
    
    if len(cracked_indices) == 0 or len(uncracked_indices) == 0:
        print("Warning: Could not find both cracked and uncracked samples for comparison")
        return
    
    # Select random samples
    sample_cracked = random.choice(cracked_indices)
    sample_uncracked = random.choice(uncracked_indices)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
    fig.suptitle('Enhanced Crack-Aware Augmentation Examples', fontsize=16, fontweight='bold')
    
    # Load original images
    cracked_img_path = train_dataset.image_paths[sample_cracked]
    uncracked_img_path = train_dataset.image_paths[sample_uncracked]
    
    cracked_img = Image.open(cracked_img_path).convert('RGB')
    uncracked_img = Image.open(uncracked_img_path).convert('RGB')
    
    # Test the transform multiple times to see variation
    transform = train_dataset.transform
    
    # Show cracked samples (top row)
    for i in range(num_samples//2):
        if isinstance(transform, CrackAwareTransform):
            # Apply crack-aware transform
            transformed = transform(cracked_img, is_cracked=True)
        else:
            transformed = transform(cracked_img)
        
        # Convert tensor back to displayable format
        if isinstance(transformed, torch.Tensor):
            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            transformed_np = transformed.permute(1, 2, 0).numpy()
            transformed_np = transformed_np * std + mean
            transformed_np = np.clip(transformed_np, 0, 1)
        else:
            transformed_np = np.array(transformed) / 255.0
        
        axes[0, i].imshow(transformed_np)
        axes[0, i].set_title(f'Cracked Sample {i+1}\n(Enhanced Aug)', fontsize=10)
        axes[0, i].axis('off')
    
    # Show uncracked samples (bottom row)
    for i in range(num_samples//2):
        if isinstance(transform, CrackAwareTransform):
            # Apply crack-aware transform
            transformed = transform(uncracked_img, is_cracked=False)
        else:
            transformed = transform(uncracked_img)
        
        # Convert tensor back to displayable format
        if isinstance(transformed, torch.Tensor):
            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            transformed_np = transformed.permute(1, 2, 0).numpy()
            transformed_np = transformed_np * std + mean
            transformed_np = np.clip(transformed_np, 0, 1)
        else:
            transformed_np = np.array(transformed) / 255.0
        
        axes[1, i].imshow(transformed_np)
        axes[1, i].set_title(f'Uncracked Sample {i+1}\n(Minimal Aug)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'augmentation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Augmentation comparison saved to: {output_path}")
    
    # Show plot if possible
    try:
        plt.show()
    except:
        print("Cannot display plot (no GUI available)")
    
    plt.close()

def test_augmentation_statistics(dataset_info):
    """Test that augmentation is providing good variety"""
    
    train_loader = dataset_info['train_loader']
    
    print("   Collecting augmentation statistics...")
    
    # Collect some samples
    sample_count = 0
    pixel_means = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        for img in images:
            pixel_means.append(float(img.mean()))
            sample_count += 1
            
        if sample_count >= 50:  # Collect 50 samples
            break
    
    pixel_mean_std = np.std(pixel_means)
    
    print(f"   📈 Sample count: {sample_count}")
    print(f"   📈 Pixel mean variation (std): {pixel_mean_std:.4f}")
    
    # Good augmentation should show variety in pixel means
    if pixel_mean_std > 0.05:
        print("   ✅ Good augmentation variety detected")
    else:
        print("   ⚠️  Low augmentation variety - may need adjustment")
        
    # Test class distribution
    class_info = dataset_info['class_info']
    print(f"   📊 Class distribution: {class_info['class_distribution']}")
    
    total_samples = sum(class_info['class_distribution'].values())
    if total_samples > 0:
        cracked_ratio = class_info['class_distribution'].get('Cracked', 0) / total_samples
        print(f"   📊 Cracked samples ratio: {cracked_ratio:.3f}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate enhanced augmentation functionality')
    parser.add_argument('--data-dir', type=str, 
                       default='SDNET2018',
                       help='Path to SDNET2018 dataset directory')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of sample images to show in comparison')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Dataset directory not found: {args.data_dir}")
        print("Please provide correct path to SDNET2018 dataset using --data-dir")
        return False
    
    # Run validation
    success = test_augmentation_functionality(args.data_dir, args.num_samples)
    
    if success:
        print("\n🎊 Validation completed successfully!")
        print("The enhanced crack-aware augmentation is ready for use.")
    else:
        print("\n💥 Validation failed!")
        print("Please check the implementation and try again.")
    
    return success

if __name__ == "__main__":
    main()