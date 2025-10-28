# manual_salun.py
import torch
import os
import json
import argparse
from datetime import datetime
from utils import get_dataset, get_model, compute_gradients

def manual_thresholding(gradients, retention_rate):
    """
    Original SalUn manual thresholding using quantiles
    """
    # Flatten all gradients
    all_grads = []
    for grad in gradients.values():
        all_grads.append(grad.view(-1))
    
    all_grads = torch.cat(all_grads)
    
    # Compute threshold based on retention rate
    threshold = torch.quantile(all_grads.abs(), retention_rate)
    
    # Create mask
    mask = {}
    for name, grad in gradients.items():
        mask[name] = (grad.abs() > threshold).float()
    
    actual_retention = sum([m.sum().item() for m in mask.values()]) / sum([m.numel() for m in mask.values()])
    print(f"📏 Manual thresholding: Target={retention_rate:.1%}, Actual={actual_retention:.1%}")
    
    return mask, actual_retention

def run_manual_salun(forget_percent, method, retention_rate, model_path, dataset='cifar100', device='cuda'):
    """
    Run manual SalUn with fixed retention rate
    """
    print(f"🔧 Manual SalUn: {method}, Forget {forget_percent}%, Retention {retention_rate:.1%}")
    
    # Get data and model
    train_loader, test_loader, retain_loader, forget_loader, _ = get_dataset(
        dataset=dataset, 
        batch_size=128, 
        forget_percent=forget_percent
    )
    
    model = get_model(dataset=dataset, model_path=model_path, device=device)
    
    # Compute gradients based on method
    if method == 'FT':
        gradients = compute_gradients(model, retain_loader, device, method='FT')
    elif method == 'RL': 
        gradients = compute_gradients(model, retain_loader, device, method='RL')
    elif method == 'GA':
        gradients = compute_gradients(model, retain_loader, forget_loader, device, method='GA')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply manual thresholding
    mask, actual_retention = manual_thresholding(gradients, retention_rate)
    
    # Save mask
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_dir = f"results/baselines/manual_masks/{method}_{forget_percent}_{int(retention_rate*100)}"
    os.makedirs(mask_dir, exist_ok=True)
    
    torch.save(mask, os.path.join(mask_dir, f'mask_{timestamp}.pt'))
    
    # Run unlearning with this mask (you'll need to adapt main_forget.py)
    # This would call your existing unlearning code with the manual mask
    
    results = {
        'forget_percent': forget_percent,
        'method': f'manual_{method}',
        'retention_rate': retention_rate,
        'actual_retention': actual_retention,
        'mask_path': mask_dir
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Manual SalUn baseline')
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--methods', nargs='+', type=str, 
                       default=['FT', 'RL'], help='Methods to run')
    parser.add_argument('--retention_rates', nargs='+', type=float,
                       default=[0.3, 0.5, 0.7], help='Retention rates to test')
    parser.add_argument('--percentages', nargs='+', type=int,
                       default=[10, 50, 90], help='Forgetting percentages')
    parser.add_argument('--device', default='cuda', type=str)
    
    args = parser.parse_args()
    
    all_results = []
    for method in args.methods:
        for retention_rate in args.retention_rates:
            for percent in args.percentages:
                results = run_manual_salun(
                    forget_percent=percent,
                    method=method,
                    retention_rate=retention_rate,
                    model_path=args.model_path,
                    dataset=args.dataset,
                    device=args.device
                )
                all_results.append(results)
    
    # Save all results
    output_dir = "results/baselines/manual_salun"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'manual_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✅ Manual SalUn completed: {len(all_results)} experiments")
    print("Results saved to:", os.path.join(output_dir, 'manual_results.json'))

if __name__ == '__main__':
    main()