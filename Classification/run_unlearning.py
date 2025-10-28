import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import arg_parser
from utils import setup_seed, setup_model_dataset, save_checkpoint
from trainer import validate

def run_unlearning_for_percentage(args, forget_percentage, method, mask_path):
    """
    Run unlearning for specific percentage and method
    """
    print(f"\n🎯 Running unlearning for {forget_percentage*100:.0f}% forgetting with {method}")
    
    # Setup device
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(args.gpu))
    
    # Create save directory
    save_dir = f"unlearn_{method}_{int(forget_percentage*100)}percent"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model, train_loader, val_loader, test_loader, marked_loader = setup_model_dataset(args)
    model = model.to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    
    # Load mask
    mask = torch.load(mask_path)
    total_params = sum(mask_tensor.numel() for mask_tensor in mask.values())
    retained_params = sum(mask_tensor.sum().item() for mask_tensor in mask.values())
    retention_rate = retained_params / total_params * 100
    print(f"🎭 Loaded mask: {retention_rate:.1f}% retention")

    # Mark dataset for this percentage
    def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
        np.random.seed(seed)
        targets = np.array(dataset.targets)
        total_samples = len(dataset)
        num_to_forget = int(forget_percentage * total_samples)
        forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False)
        
        for idx in forget_indices:
            dataset.targets[idx] = -dataset.targets[idx] - 1
        
        return dataset

    def create_forget_retain_loaders(train_loader_full, forget_percentage, args):
        def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
            setup_seed(seed)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=shuffle,
            )

        original_dataset = copy.deepcopy(train_loader_full.dataset)
        marked_dataset = mark_dataset_for_percentage(original_dataset, forget_percentage, seed=args.seed)
        
        forget_dataset = copy.deepcopy(marked_dataset)
        retain_dataset = copy.deepcopy(marked_dataset)
        
        targets = np.array(forget_dataset.targets)
        forget_mask = targets < 0
        retain_mask = targets >= 0
        
        forget_dataset.data = forget_dataset.data[forget_mask]
        forget_dataset.targets = (-targets[forget_mask] - 1).tolist()
        
        retain_dataset.data = retain_dataset.data[retain_mask]
        retain_dataset.targets = targets[retain_mask].tolist()
        
        forget_loader = replace_loader_dataset(forget_dataset, seed=args.seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=args.seed, shuffle=True)

        print(f"📊 Dataset: {len(forget_dataset)} forget, {len(retain_dataset)} retain samples")
        
        return forget_loader, retain_loader

    # Create loaders
    forget_loader, retain_loader = create_forget_retain_loaders(train_loader, forget_percentage, args)
    
    # Choose loader based on method
    if method == "GA":
        train_loader = forget_loader
        print("🔧 Unlearning Method: GA (Gradient Ascent on forget set)")
    elif method == "FT":
        train_loader = retain_loader
        print("🔧 Unlearning Method: FT (Fine-tuning on retain set)")
    elif method == "RL":
        train_loader = retain_loader
        print("🔧 Unlearning Method: RL (Retain Learning)")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # Method-specific learning rates
    lr_dict = {'GA': 0.001, 'FT': 0.1, 'RL': 0.1}
    unlearn_lr = lr_dict.get(method, 0.1)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler()
    
    # Store initial state for mask verification
    initial_masked_state = {}
    for name, param in model.named_parameters():
        if name in mask:
            initial_masked_state[name] = param.data.clone()

    # Training loop
    best_val_acc = 0
    results = {
        'train_acc': [],
        'val_acc': [],
        'final_val_acc': 0,
        'best_val_acc': 0,
        'forget_percentage': forget_percentage,
        'method': method,
        'retention_rate': retention_rate
    }

    for epoch in range(args.unlearn_epochs):
        model.train()
        running_corrects = 0
        total_samples = 0
        
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Store current state for restoration
            batch_initial_state = {}
            for name, param in model.named_parameters():
                if name in mask:
                    batch_initial_state[name] = param.data.clone()

            optimizer.zero_grad()

            # Forward pass
            with autocast():
                outputs = model(images)
                if method == "GA":
                    loss = -criterion(outputs, target)  # Gradient ascent
                else:
                    loss = criterion(outputs, target)   # Gradient descent

            # Backward pass
            scaler.scale(loss).backward()
            
            # Apply mask to gradients
            for name, param in model.named_parameters():
                if name in mask and param.grad is not None:
                    mask_tensor = mask[name].cuda()
                    param.grad.data *= mask_tensor

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Restore masked parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in mask and name in batch_initial_state:
                        restoration_mask = mask[name].cuda()
                        param.data = param.data * restoration_mask + batch_initial_state[name] * (1 - restoration_mask)

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == target).item()
            total_samples += images.size(0)

        # Epoch statistics
        train_acc = running_corrects / total_samples if total_samples > 0 else 0.0
        val_acc = validate(val_loader, model, criterion, args, use_amp=True, scaler=scaler)
        
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"🎯 Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    results['final_val_acc'] = val_acc
    results['best_val_acc'] = best_val_acc
    
    # Save results
    results_path = os.path.join(save_dir, f"results_{method}_{int(forget_percentage*100)}percent.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final model
    model_path = os.path.join(save_dir, f"model_{method}_{int(forget_percentage*100)}percent.pth")
    torch.save(model.state_dict(), model_path)
    
    print(f"💾 Results saved to: {results_path}")
    print(f"💾 Model saved to: {model_path}")
    
    return results

def main():
    args = arg_parser.parse_args()
    
    # Define experiments
    forget_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    methods = ['GA', 'FT', 'RL']
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING UNLEARNING FOR METHOD: {method}")
        print(f"{'='*60}")
        
        method_results = {}
        
        for percentage in forget_percentages:
            # Construct mask path
            mask_dir = f"masks_{method}_{int(percentage*100)}percent"
            mask_path = os.path.join(mask_dir, f"mask_otsu_{method}_{int(percentage*100)}percent.pt")
            
            if not os.path.exists(mask_path):
                print(f"❌ Mask not found: {mask_path}")
                print(f"   Please generate masks first using generate_masks.py")
                continue
            
            try:
                results = run_unlearning_for_percentage(args, percentage, method, mask_path)
                method_results[percentage] = results
            except Exception as e:
                print(f"❌ Error running {method} at {percentage*100:.0f}%: {e}")
                method_results[percentage] = {'error': str(e)}
        
        all_results[method] = method_results
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"📊 UNLEARNING COMPLETE SUMMARY")
    print(f"{'='*80}")
    
    for method in methods:
        print(f"\n{method} Method:")
        print("Percentage | Final Val Acc | Best Val Acc | Retention")
        print("-" * 65)
        for percentage in forget_percentages:
            if percentage in all_results[method]:
                results = all_results[method][percentage]
                if 'error' in results:
                    print(f"{percentage*100:>4.0f}%     | {'ERROR':>12} | {'ERROR':>11} | {'ERROR':>9}")
                else:
                    print(f"{percentage*100:>4.0f}%     | {results['final_val_acc']:>12.2f} | {results['best_val_acc']:>11.2f} | {results['retention_rate']:>8.1f}%")

if __name__ == "__main__":
    main()