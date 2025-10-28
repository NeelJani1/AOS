# evaluate_all_mia.py
import os
import json
import torch
import argparse
from datetime import datetime
import glob
from mia_evaluation import MIA, SVC_MIA
from utils import setup_model_dataset, replace_loader_dataset
from main_generate_masks import mark_dataset_for_percentage, split_marked_dataset

def load_unlearned_model(model_path, args, device):
    """Load unlearned model"""
    model, _, _, _, _ = setup_model_dataset(args)
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def evaluate_all_mia(args):
    """Evaluate MIA for all methods and percentages"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(args.output_base_dir, f"{args.experiment_pattern}*"))
    
    all_mia_results = {}
    
    for exp_dir in experiment_dirs:
        print(f"\n🔍 Processing experiment: {exp_dir}")
        
        # Find all unlearning result directories
        unlearning_dirs = glob.glob(os.path.join(exp_dir, 'unlearning', '*', '*percent'))
        
        for unlearn_dir in unlearning_dirs:
            method = os.path.basename(os.path.dirname(unlearn_dir))
            percentage = int(os.path.basename(unlearn_dir).replace('percent', '')) / 100.0
            
            print(f"📊 Evaluating {method} - {percentage*100:.0f}%...")
            
            # Load model
            model_path = os.path.join(unlearn_dir, 'model_SA_best.pth.tar')
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                continue
            
            model = load_unlearned_model(model_path, args, device)
            
            # Create data loaders for MIA
            original_model, train_loader_full, val_loader, _, _ = setup_model_dataset(args)
            
            # Mark and split dataset
            original_dataset = train_loader_full.dataset
            marked_dataset = mark_dataset_for_percentage(original_dataset, percentage, seed=args.seed)
            forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
            
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, shuffle=False)
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, shuffle=False)
            
            # Run MIA evaluation
            try:
                mia_results = MIA(
                    retain_loader_train=retain_loader,
                    retain_loader_test=retain_loader,
                    forget_loader=forget_loader,
                    test_loader=val_loader,
                    model=model,
                    device=device
                )
                
                svc_mia_results = SVC_MIA(
                    shadow_train=retain_loader,
                    target_train=retain_loader,
                    target_test=forget_loader,
                    shadow_test=val_loader,
                    model=model
                )
                
                # Store results
                key = f"{method}_{int(percentage*100)}percent"
                all_mia_results[key] = {
                    'mia': mia_results,
                    'svc_mia': svc_mia_results,
                    'method': method,
                    'percentage': percentage
                }
                
                print(f"✅ MIA completed: {method} - {percentage*100:.0f}%")
                
            except Exception as e:
                print(f"❌ MIA failed for {method} - {percentage*100:.0f}%: {e}")
    
    # Save comprehensive MIA results
    output_dir = "results/mia_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'mia_results_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(all_mia_results, f, indent=4)
    
    print(f"\n📊 MIA Evaluation Summary:")
    print("Method | Percentage | MIA Acc | SVC MIA Acc")
    print("-" * 50)
    
    for key, results in all_mia_results.items():
        mia_acc = results['mia'].get('confidence', [0, 0])
        svc_mia_acc = results['svc_mia'].get('confidence', 0)
        
        if isinstance(mia_acc, tuple):
            mia_acc = sum(mia_acc) / 2  # Average train and test accuracy
        
        print(f"{results['method']:6} | {results['percentage']*100:>4.0f}%     | {mia_acc:.3f}    | {svc_mia_acc:.3f}")
    
    return all_mia_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate MIA for all unlearning methods')
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--experiment_pattern', default='quick_fix', type=str,
                       help='Pattern to match experiment directories')
    parser.add_argument('--output_base_dir', default='results', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    
    args = parser.parse_args()
    evaluate_all_mia(args)

if __name__ == '__main__':
    main()