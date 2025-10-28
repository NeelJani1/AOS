import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os

def analyze_unlearning_performance(results_dir):
    """Analyze and visualize unlearning performance"""
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "**", "unlearning_results.json"), recursive=True)
    
    data = []
    for file in result_files:
        with open(file, 'r') as f:
            result = json.load(f)
            data.append({
                'method': result['method'],
                'percentage': result['forget_percentage'] * 100,
                'test_accuracy': result['final_test_accuracy'],
                'retain_accuracy': result['final_retain_accuracy'],
                'forget_accuracy': result['final_forget_accuracy'],
                'mask_retention': result['mask_retention'],
                'file_path': file
            })
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Test Accuracy vs Forgetting Percentage
    plt.subplot(2, 2, 1)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['percentage'], method_data['test_accuracy'], 
                'o-', label=method, markersize=6, linewidth=2)
    plt.xlabel('Forget Percentage (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Forgetting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Forget Accuracy vs Forgetting Percentage
    plt.subplot(2, 2, 2)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['percentage'], method_data['forget_accuracy'], 
                's-', label=method, markersize=6, linewidth=2)
    plt.xlabel('Forget Percentage (%)')
    plt.ylabel('Forget Accuracy (%)')
    plt.title('Forgetting Effectiveness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Retention vs Forgetting Trade-off
    plt.subplot(2, 2, 3)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.scatter(method_data['retain_accuracy'], method_data['forget_accuracy'],
                   label=method, s=80, alpha=0.7)
    plt.xlabel('Retain Accuracy (%)')
    plt.ylabel('Forget Accuracy (%)')
    plt.title('Retention vs Forgetting Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Mask Retention vs Performance
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(df['mask_retention'], df['test_accuracy'], 
                         c=df['percentage'], cmap='viridis', s=80, alpha=0.7)
    plt.colorbar(scatter, label='Forget Percentage (%)')
    plt.xlabel('Mask Retention (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Mask Retention vs Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n📊 PERFORMANCE SUMMARY:")
    summary = df.groupby('method').agg({
        'test_accuracy': ['mean', 'std'],
        'retain_accuracy': ['mean', 'std'],
        'forget_accuracy': ['mean', 'std']
    }).round(2)
    
    print(summary)
    
    return df

# Usage
if __name__ == '__main__':
    results_df = analyze_unlearning_performance("results/comprehensive_unlearning_20251026_022500")