import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob

def parse_log_file(log_path):
    """Parse log file to extract final accuracies"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for validation accuracy patterns
        test_acc = None
        best_val_acc = None
        
        # Pattern 1: Look for "valid_accuracy XX.XX" 
        if 'valid_accuracy' in content:
            lines = content.split('\n')
            for line in lines:
                if 'valid_accuracy' in line and 'valid_accuracy_curve' not in line:
                    try:
                        acc_str = line.split('valid_accuracy')[-1].strip()
                        test_acc = float(acc_str)
                        break
                    except:
                        continue
        
        # Pattern 2: Look for best validation accuracy
        if 'Best validation accuracy:' in content:
            lines = content.split('\n')
            for line in lines:
                if 'Best validation accuracy:' in line:
                    try:
                        acc_str = line.split('Best validation accuracy:')[-1].strip().split()[0]
                        best_val_acc = float(acc_str)
                        break
                    except:
                        continue
        
        # Use best_val_acc if test_acc not found
        if test_acc is None and best_val_acc is not None:
            test_acc = best_val_acc
        
        return {
            'test_acc': test_acc,
            'best_val_acc': best_val_acc
        }
    except Exception as e:
        print(f"❌ Error parsing {log_path}: {e}")
        return None

def load_experiment_results(base_dir):
    """Load all experiment results into a structured format"""
    results = []
    methods = ['GA', 'FT', 'RL']
    
    print("🔍 Scanning for completed experiments...")
    
    for method in methods:
        method_dir = os.path.join(base_dir, 'unlearning', method)
        if not os.path.exists(method_dir):
            print(f"⚠️  Method directory not found: {method_dir}")
            continue
            
        # Find all experiment directories for this method
        exp_pattern = os.path.join(method_dir, '*')
        exp_dirs = glob.glob(exp_pattern)
        
        for exp_dir in exp_dirs:
            if not os.path.isdir(exp_dir):
                continue
                
            # Extract percentage from directory name
            dir_name = os.path.basename(exp_dir)
            print(f"📁 Checking directory: {dir_name}")
            
            # Try different patterns to extract percentage
            percent = None
            for part in dir_name.split('_'):
                if 'percent' in part:
                    try:
                        percent = int(part.replace('percent', ''))
                        break
                    except:
                        continue
            
            if percent is None:
                print(f"   ⚠️  Could not extract percentage from {dir_name}")
                continue
            
            # Check for model files and logs - FIXED: looking for correct file names
            model_files = glob.glob(os.path.join(exp_dir, 'model_SA_best.pth.tar'))
            log_path = os.path.join(exp_dir, 'log.txt')
            
            print(f"   🔍 Model files: {len(model_files)}, Log: {os.path.exists(log_path)}")
            
            if model_files and os.path.exists(log_path):
                # Parse log file
                accuracies = parse_log_file(log_path)
                
                if accuracies and accuracies['test_acc'] is not None:
                    results.append({
                        'method': method,
                        'percentage': percent,
                        'test_acc': accuracies['test_acc'],
                        'best_val_acc': accuracies['best_val_acc'],
                        'completed': True,
                        'model_path': model_files[0],
                        'log_path': log_path
                    })
                    print(f"   ✅ Loaded: {method} {percent}% - Test Acc: {accuracies['test_acc']}")
                else:
                    print(f"   ⚠️  No accuracy data found in log")
                    results.append({
                        'method': method,
                        'percentage': percent,
                        'test_acc': None,
                        'completed': False
                    })
            else:
                results.append({
                    'method': method,
                    'percentage': percent,
                    'test_acc': None,
                    'completed': False
                })
    
    return pd.DataFrame(results)

def analyze_method_performance(df):
    """Analyze performance of each method"""
    print("\n" + "="*60)
    print("📊 ADAPTIVE OTSU UNLEARNING PERFORMANCE ANALYSIS")
    print("="*60)
    
    completed_df = df[df['completed'] == True]
    
    if completed_df.empty:
        print("❌ No completed experiments found!")
        return
    
    print(f"📈 Found {len(completed_df)} completed experiments")
    
    # Method-wise analysis
    for method in ['GA', 'FT', 'RL']:
        method_df = completed_df[completed_df['method'] == method]
        if len(method_df) > 0:
            print(f"\n🎯 {method} METHOD PERFORMANCE:")
            print(f"   Completed experiments: {len(method_df)}")
            
            # Calculate averages
            avg_test_acc = method_df['test_acc'].mean()
            min_test_acc = method_df['test_acc'].min()
            max_test_acc = method_df['test_acc'].max()
            
            print(f"   Test Accuracy: {avg_test_acc:.2f}% (min: {min_test_acc:.2f}%, max: {max_test_acc:.2f}%)")
            
            # Show performance by percentage
            print(f"   Performance by percentage:")
            for percent in sorted(method_df['percentage'].unique()):
                percent_df = method_df[method_df['percentage'] == percent]
                if len(percent_df) > 0:
                    acc = percent_df['test_acc'].iloc[0]
                    print(f"     {percent}%: {acc:.2f}%")

def create_simple_plot(df, output_dir):
    """Create a simple plot for available data"""
    completed_df = df[df['completed'] == True]
    
    if completed_df.empty:
        print("❌ No data to plot!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot Test Accuracy vs Forgetting Percentage
    for method in completed_df['method'].unique():
        method_df = completed_df[completed_df['method'] == method].sort_values('percentage')
        if len(method_df) > 0:
            plt.plot(method_df['percentage'], method_df['test_acc'], 
                     marker='o', linewidth=3, markersize=8, label=method)
    
    plt.title('Adaptive Otsu Unlearning - Test Accuracy vs Forgetting Percentage', fontweight='bold', fontsize=14)
    plt.xlabel('Forgetting Percentage (%)', fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 101, 10))
    
    plot_path = os.path.join(output_dir, 'unlearning_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Performance plot saved to: {plot_path}")

def main():
    base_dir = "/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification/results/adaptive_otsu_fixed_all_20251026_022500"
    output_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Loading experiment results...")
    df = load_experiment_results(base_dir)
    
    if df.empty:
        print("❌ No experiment data found!")
        return
    
    print(f"\n📊 Loaded {len(df)} experiment records")
    print(f"✅ Completed experiments: {len(df[df['completed'] == True])}")
    
    # Save raw data
    csv_path = os.path.join(output_dir, 'unlearning_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Raw data saved to: {csv_path}")
    
    # Performance analysis
    analyze_method_performance(df)
    
    # Create visualization
    create_simple_plot(df, output_dir)
    
    print("\n🎉 EVALUATION COMPLETE!")
    print(f"📁 All results saved in: {output_dir}")

if __name__ == "__main__":
    main()
