#!/usr/bin/env python3
"""
Experiment Runner for SNN Crack Detection Evaluation
Phase 1 Implementation - CrackVision-inspired evaluation
"""

import subprocess
import sys
import os
from datetime import datetime

def run_experiment(experiment_type, additional_args=None):
    """Run a specific experiment type"""
    
    base_cmd = [
        sys.executable, "sdnet_spiking.py",
        "--data-dir", "/home/duyanh/Workspace/SDNET_spiking/SDNET2018",
        "--batch-size", "8",
        "--time-steps", "10",
        "--num-epochs", "20",
        "--learning-rate", "0.001"
    ]
    
    if additional_args:
        base_cmd.extend(additional_args)
    
    print(f"\n{'='*60}")
    print(f"Running {experiment_type.upper()} Experiment")
    print(f"Command: {' '.join(base_cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(base_cmd, check=True, capture_output=False)
        print(f"\n✅ {experiment_type} experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {experiment_type} experiment failed with error: {e}")
        return False

def main():
    """Run Phase 1 evaluation experiments"""
    
    print("🧠 SNN Crack Detection - Phase 1 Evaluation")
    print("=" * 60)
    print("CrackVision-inspired comprehensive evaluation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    experiments = [
        {
            "name": "Cross-Validation",
            "description": "5-fold stratified cross-validation",
            "args": ["--eval-mode", "cross_validation", "--cv-folds", "5"]
        },
        {
            "name": "Baseline Comparison", 
            "description": "Compare SNN vs CNN baselines (ResNet50, etc.)",
            "args": ["--eval-mode", "baseline_comparison"]
        },
        {
            "name": "Comprehensive Evaluation",
            "description": "Combined CV + baseline comparison",
            "args": ["--eval-mode", "comprehensive"]
        }
    ]
    
    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}: {exp['description']}")
    
    print("\n0. Run all experiments")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect experiment to run (0-3, q): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye! 👋")
            break
            
        elif choice == '0':
            print("\n🚀 Running all experiments...")
            success_count = 0
            
            for exp in experiments:
                if run_experiment(exp['name'], exp['args']):
                    success_count += 1
            
            print(f"\n📊 Summary: {success_count}/{len(experiments)} experiments completed successfully")
            break
            
        elif choice in ['1', '2', '3']:
            exp_idx = int(choice) - 1
            exp = experiments[exp_idx]
            
            print(f"\n🎯 Running {exp['name']} experiment...")
            run_experiment(exp['name'], exp['args'])
            break
            
        else:
            print("❌ Invalid choice. Please select 0-3 or q.")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def quick_test():
    """Run a quick test with minimal epochs"""
    print("🧪 Running quick test...")
    
    test_args = [
        "--eval-mode", "baseline_comparison",
        "--num-epochs", "2",
        "--batch-size", "4"
    ]
    
    run_experiment("Quick Test", test_args)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        main()