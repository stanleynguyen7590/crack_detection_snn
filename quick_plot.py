#!/usr/bin/env python3
"""
Quick Plot Script - Simplified interface for manual plotting
"""

import os
import sys
import subprocess
from pathlib import Path

def find_models(directory="checkpoints"):
    """Find all model files in directory"""
    if not os.path.exists(directory):
        return []
    
    model_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    return model_files

def find_results(directory="results"):
    """Find all result files in directory"""
    if not os.path.exists(directory):
        return []
    
    result_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                result_files.append(os.path.join(root, file))
    return result_files

def interactive_model_selection():
    """Interactive model selection"""
    print("🔍 Searching for model files...")
    models = find_models()
    
    if not models:
        print("❌ No model files found in 'checkpoints/' directory")
        return None
    
    print(f"\n📁 Found {len(models)} model file(s):")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def interactive_results_selection():
    """Interactive results selection"""
    print("🔍 Searching for result files...")
    results = find_results()
    
    if not results:
        print("❌ No result files found in 'results/' directory")
        return None
    
    print(f"\n📁 Found {len(results)} result file(s):")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result}")
    
    while True:
        try:
            choice = input(f"\nSelect results (1-{len(results)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                return results[idx]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def get_model_type():
    """Get model type from user"""
    print("\n🤖 Model types:")
    print("  1. SNN (Spiking Neural Network)")
    print("  2. ResNet50")
    print("  3. ResNet18") 
    print("  4. Xception-style")
    
    types = ['snn', 'resnet50', 'resnet18', 'xception']
    
    while True:
        try:
            choice = input("Select model type (1-4): ")
            idx = int(choice) - 1
            if 0 <= idx < len(types):
                return types[idx]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def main():
    """Interactive main function"""
    print("🎨 Quick Plot Tool for SDNET SNN Project")
    print("=" * 50)
    
    print("\n📊 What would you like to plot?")
    print("  1. Analyze single model")
    print("  2. Compare multiple models")
    print("  3. Plot from saved results")
    print("  4. Plot training history")
    print("  5. Quick analysis (best model)")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ")
            break
        except ValueError:
            print("Please enter a number.")
    
    # Create output directory
    output_dir = "quick_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if choice == "1":
        # Single model analysis
        print("\n🔍 Single Model Analysis")
        model_path = interactive_model_selection()
        if not model_path:
            return
        
        model_type = get_model_type()
        model_name = input("Enter model name (or press Enter for default): ").strip()
        if not model_name:
            model_name = Path(model_path).stem
        
        cmd = [
            "python", "manual_plotting.py",
            "--mode", "single",
            "--model-path", model_path,
            "--model-type", model_type,
            "--model-name", model_name,
            "--output-dir", output_dir
        ]
        
        # Check if SDNET2018 exists in different locations
        data_dirs = ["./SDNET2018", "../SDNET2018", "SDNET2018"]
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                cmd.extend(["--data-dir", data_dir])
                break
        
        print(f"\n🚀 Running analysis...")
        subprocess.run(cmd)
        
    elif choice == "2":
        # Multiple model comparison
        print("\n📊 Multiple Model Comparison")
        
        if os.path.exists("example_model_config.json"):
            use_config = input("Use example_model_config.json? (y/n): ").lower().startswith('y')
            if use_config:
                cmd = [
                    "python", "manual_plotting.py",
                    "--mode", "compare",
                    "--config-file", "example_model_config.json",
                    "--output-dir", output_dir
                ]
                
                # Check for data directory
                data_dirs = ["./SDNET2018", "../SDNET2018", "SDNET2018"]
                for data_dir in data_dirs:
                    if os.path.exists(data_dir):
                        cmd.extend(["--data-dir", data_dir])
                        break
                
                print(f"\n🚀 Running comparison...")
                subprocess.run(cmd)
                return
        
        print("❌ No model configuration found. Please create example_model_config.json first.")
        
    elif choice == "3":
        # Plot from results
        print("\n📈 Plot from Saved Results")
        results_path = interactive_results_selection()
        if not results_path:
            return
        
        cmd = [
            "python", "manual_plotting.py",
            "--mode", "results",
            "--results-file", results_path,
            "--output-dir", output_dir
        ]
        
        print(f"\n🚀 Generating plots from results...")
        subprocess.run(cmd)
        
    elif choice == "4":
        # Training history
        print("\n📉 Plot Training History")
        
        # Look for training history files
        history_files = []
        for ext in ['*.json', '*.pkl']:
            history_files.extend(Path('.').glob(f"*history*{ext}"))
            history_files.extend(Path('.').glob(f"*train*{ext}"))
        
        if not history_files:
            print("❌ No training history files found")
            return
        
        print(f"\nFound {len(history_files)} history file(s):")
        for i, file in enumerate(history_files):
            print(f"  {i+1}. {file}")
        
        while True:
            try:
                choice = input(f"Select file (1-{len(history_files)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(history_files):
                    history_file = str(history_files[idx])
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        model_name = input("Enter model name for plots: ").strip() or "model"
        
        cmd = [
            "python", "manual_plotting.py",
            "--mode", "history",
            "--history-file", history_file,
            "--model-name", model_name,
            "--output-dir", output_dir
        ]
        
        print(f"\n🚀 Plotting training history...")
        subprocess.run(cmd)
        
    elif choice == "5":
        # Quick analysis of best model
        print("\n⚡ Quick Analysis (Best Model)")
        
        # Look for best model
        best_model_paths = [
            "checkpoints/best_model.pth",
            "best_model.pth",
            "checkpoints/final_model.pth",
            "final_model.pth"
        ]
        
        best_model = None
        for path in best_model_paths:
            if os.path.exists(path):
                best_model = path
                break
        
        if not best_model:
            print("❌ No best model found. Looking for any model...")
            models = find_models()
            if models:
                best_model = models[0]
                print(f"📁 Using: {best_model}")
            else:
                print("❌ No models found")
                return
        else:
            print(f"📁 Found best model: {best_model}")
        
        cmd = [
            "python", "manual_plotting.py",
            "--mode", "single",
            "--model-path", best_model,
            "--model-type", "snn",
            "--model-name", "Best_SNN_Model",
            "--output-dir", output_dir
        ]
        
        # Check for data directory
        data_dirs = ["./SDNET2018", "../SDNET2018", "SDNET2018"]
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                cmd.extend(["--data-dir", data_dir])
                break
        
        print(f"\n🚀 Running quick analysis...")
        subprocess.run(cmd)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Check results in: {output_dir}/")
    
    # Open output directory if possible
    if sys.platform.startswith('darwin'):  # macOS
        subprocess.run(['open', output_dir])
    elif sys.platform.startswith('linux'):  # Linux
        subprocess.run(['xdg-open', output_dir])
    elif sys.platform.startswith('win'):  # Windows
        subprocess.run(['explorer', output_dir])

if __name__ == "__main__":
    main()