# evaluate_completed_results.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

def load_all_results(results_base_dir="results"):
    """
    Load all results from organized directory structure
    """
    all_results = []
    
    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(results_base_dir, "adaptive_otsu_*"))
    
    for exp_dir in experiment_dirs:
        print(f"📁 Processing: {exp_dir}")
        
        # Look for unlearning results
        unlearning_dir = os.path.join(exp_dir, "unlearning")
        if not os.path.exists(unlearning_dir):
            continue
            
        # Process each method
        for method in ['GA', 'FT', 'RL']:
            method_dir = os.path.join(unlearning_dir, method)
            if not os.path.exists(method_dir):
                continue
                
            # Process each percentage
            for percentage_dir in glob.glob(os.path.join(method_dir, "*")):
                if not os.path.isdir(percentage_dir):
                    continue
                    
                # Load results JSON
                results_path = os.path.join(percentage_dir, "unlearning_results.json")
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        results['experiment_dir'] = exp_dir
                        all_results.append(results)
    
    return all_results

def create_comprehensive_analysis(results, output_dir):
    """
    Create comprehensive analysis and visualizations
    """
    if not results:
        print("❌ No results to analyze")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create summary statistics
    summary = df.groupby(['method', 'forget_percentage']).agg({
        'final_test_accuracy': ['mean', 'std'],
        'final_retain_accuracy': ['mean', 'std'],
        'final_forget_accuracy': ['mean', 'std']
    }).round(2)
    
    # Create visualizations
    plt.figure(figsize(15, 10))
    
    # Plot 1: Test Accuracy vs Forget Percentage by Method
    plt.subplot(2, 2, 1)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['forget_percentage']*100, method_data['final_test_accuracy'], 
                'o-', label=method, markersize=6, linewidth=2)
    
    plt.xlabel('Forget Percentage (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Forgetting Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Retention Performance
    plt.subplot(2, 2, 2)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['forget_percentage']*100, method_data['final_retain_accuracy'], 
                's-', label=method, markersize=6, linewidth=2)
    
    plt.xlabel('Forget Percentage (%)')
    plt.ylabel('Retain Accuracy (%)')
    plt.title('Knowledge Retention vs Forgetting Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Forgetting Effectiveness
    plt.subplot(2, 2, 3)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['forget_percentage']*100, method_data['final_forget_accuracy'], 
                '^-', label=method, markersize=6, linewidth=2)
    
    plt.xlabel('Forget Percentage (%)')
    plt.ylabel('Forget Accuracy (%)')
    plt.title('Forgetting Effectiveness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance Trade-off
    plt.subplot(2, 2, 4)
    avg_performance = df.groupby('method').agg({
        'final_test_accuracy': 'mean',
        'final_retain_accuracy': 'mean',
        'final_forget_accuracy': 'mean'
    }).reset_index()
    
    plt.scatter(avg_performance['final_retain_accuracy'], avg_performance['final_forget_accuracy'],
               s=100, alpha=0.7)
    
    for i, row in avg_performance.iterrows():
        plt.annotate(row['method'], (row['final_retain_accuracy'], row['final_forget_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Average Retain Accuracy (%)')
    plt.ylabel('Average Forget Accuracy (%)')
    plt.title('Retention vs Forgetting Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    detailed_results_path = os.path.join(output_dir, 're-evaluated_results.csv')
    df.to_csv(detailed_results_path, index=False)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    summary.to_csv(summary_path)
    
    return df, summary

def generate_insights_report(df, output_dir):
    """
    Generate insights report with key findings
    """
    insights = []
    insights.append("# Adaptive Otsu Unlearning - Comprehensive Analysis")
    insights.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    insights.append("")
    
    # Overall performance by method
    avg_performance = df.groupby('method').agg({
        'final_test_accuracy': 'mean',
        'final_retain_accuracy': 'mean',
        'final_forget_accuracy': 'mean'
    }).round(2)
    
    insights.append("## 📊 Average Performance by Method")
    insights.append(avg_performance.to_markdown())
    insights.append("")
    
    # Key insights
    insights.append("## 🎯 Key Insights")
    
    # Best method for each metric
    best_test = avg_performance['final_test_accuracy'].idxmax()
    best_retain = avg_performance['final_retain_accuracy'].idxmax()
    best_forget = avg_performance['final_forget_accuracy'].idxmin()  # Lower is better for forgetting
    
    insights.append(f"- **Best Test Accuracy**: {best_test} ({avg_performance.loc[best_test, 'final_test_accuracy']}%)")
    insights.append(f"- **Best Knowledge Retention**: {best_retain} ({avg_performance.loc[best_retain, 'final_retain_accuracy']}%)")
    insights.append(f"- **Most Effective Forgetting**: {best_forget} ({avg_performance.loc[best_forget, 'final_forget_accuracy']}%)")
    insights.append("")
    
    # Method-specific insights
    insights.append("## ⚡ Method-Specific Insights")
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        insights.append(f"### {method} Method")
        insights.append(f"- Average Test Accuracy: {method_data['final_test_accuracy'].mean():.2f}%")
        insights.append(f"- Average Retain Accuracy: {method_data['final_retain_accuracy'].mean():.2f}%")
        insights.append(f"- Average Forget Accuracy: {method_data['final_forget_accuracy'].mean():.2f}%")
        insights.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'adaptive_otsu_insights.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(insights))
    
    return insights

def main():
    # Setup output directory
    output_dir = "results/comprehensive_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Loading all completed results...")
    
    # Load all results
    all_results = load_all_results()
    
    if not all_results:
        print("❌ No results found. Please run experiments first.")
        return
    
    print(f"📈 Found {len(all_results)} experiment results")
    
    # Create comprehensive analysis
    print("📊 Creating comprehensive analysis...")
    df, summary = create_comprehensive_analysis(all_results, output_dir)
    
    if df is not None:
        # Generate insights
        print("💡 Generating insights report...")
        insights = generate_insights_report(df, output_dir)
        
        print(f"\n✅ Comprehensive evaluation completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Analysis plot: comprehensive_analysis.png")
        print(f"📋 Detailed results: re-evaluated_results.csv")
        print(f"📊 Performance summary: performance_summary.csv")
        print(f"📝 Insights report: adaptive_otsu_insights.md")
        
        # Print key findings
        print("\n🎯 KEY FINDINGS:")
        avg_performance = df.groupby('method').agg({
            'final_test_accuracy': 'mean',
            'final_retain_accuracy': 'mean',
            'final_forget_accuracy': 'mean'
        }).round(2)
        
        for method in avg_performance.index:
            print(f"{method}: Test={avg_performance.loc[method, 'final_test_accuracy']}%, "
                  f"Retain={avg_performance.loc[method, 'final_retain_accuracy']}%, "
                  f"Forget={avg_performance.loc[method, 'final_forget_accuracy']}%")
    
    else:
        print("❌ Failed to create analysis")

if __name__ == '__main__':
    main()