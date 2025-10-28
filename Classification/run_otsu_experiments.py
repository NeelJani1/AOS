#!/usr/bin/env python3
"""
Script to run Otsu unlearning experiments with different parameters
"""

import os
import subprocess
import argparse

def run_otsu_experiment(experiment_name, otsu_method, **kwargs):
    """Run a single Otsu experiment with given parameters"""
    
    base_command = [
        "python", "main_random.py",
        "--unlearn", "RL",
        "--unlearn_epochs", "10",
        "--unlearn_lr", "0.1", 
        "--num_indexes_to_replace", "4500",
        "--model_path", "/home/neel/Unlearn-Saliency-master/saved_models/0checkpoint.pth.tar",
        "--dataset", "cifar100",
        "--batch_size", "128",
        "--otsu_method", otsu_method
    ]
    
    # Add Otsu-specific parameters
    if otsu_method == "conservative":
        base_command.extend(["--otsu_conservatism", str(kwargs.get('conservatism', 0.3))])
    elif otsu_method == "bounded":
        base_command.extend([
            "--otsu_min_retention", str(kwargs.get('min_retention', 0.3)),
            "--otsu_max_retention", str(kwargs.get('max_retention', 0.6))
        ])
    
    # Set save directory
    save_dir = f"unlearn_RL_otsu_{experiment_name}"
    base_command.extend(["--save_dir", save_dir])
    
    # Add test flag if specified
    if kwargs.get('test_methods', False):
        base_command.append("--test_otsu_methods")
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(base_command)}")
    print(f"{'='*60}")
    
    # Run the command
    try:
        subprocess.run(base_command, check=True)
        print(f"Experiment {experiment_name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_name} failed with error: {e}")
    
    return save_dir

def main():
    parser = argparse.ArgumentParser(description='Run Otsu unlearning experiments')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments')
    parser.add_argument('--test_methods', action='store_true', help='Test Otsu methods first')
    args = parser.parse_args()
    
    # Define experiments
    experiments = [
        # Basic Otsu
        {"name": "basic", "method": "basic"},
        
        # Conservative Otsu variants
        {"name": "conservative_0.1", "method": "conservative", "conservatism": 0.1},
        {"name": "conservative_0.3", "method": "conservative", "conservatism": 0.3},
        {"name": "conservative_0.5", "method": "conservative", "conservatism": 0.5},
        {"name": "conservative_0.7", "method": "conservative", "conservatism": 0.7},
        
        # Bounded Otsu variants  
        {"name": "bounded_20-50", "method": "bounded", "min_retention": 0.2, "max_retention": 0.5},
        {"name": "bounded_30-60", "method": "bounded", "min_retention": 0.3, "max_retention": 0.6},
        {"name": "bounded_40-70", "method": "bounded", "min_retention": 0.4, "max_retention": 0.7},
        
        # Layer-aware
        {"name": "layer_aware", "method": "layer_aware"},
    ]
    
    if args.run_all:
        print("Running all Otsu experiments...")
        results = []
        for exp in experiments:
            result_dir = run_otsu_experiment(**exp, test_methods=args.test_methods)
            results.append((exp['name'], result_dir))
        
        print("\n" + "="*60)
        print("Experiment Summary:")
        for name, directory in results:
            print(f"{name:20} -> {directory}")
    
    else:
        # Run just the recommended starting point
        recommended = {"name": "conservative_0.3", "method": "conservative", "conservatism": 0.3}
        run_otsu_experiment(**recommended, test_methods=True)

if __name__ == "__main__":
    main()