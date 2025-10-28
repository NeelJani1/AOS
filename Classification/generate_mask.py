import os
import copy
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import arg_parser
import utils
from otsu_utils import *
from output_utils import *
import datetime
import json

def scale_ga_gradients(gradients, scale_factor=0.5):
    """
    Scale GA gradients - MUCH MORE CONSERVATIVE NOW
    """
    print(f"🔧 Scaling GA gradients by {scale_factor}x (CONSERVATIVE)")
    with torch.no_grad():
        for name in gradients:
            gradients[name] = gradients[name] * scale_factor
    return gradients

def compute_ga_gradients_fixed(model, retain_loader, forget_loader, device):
    """
    Fixed GA gradient computation with conservative handling
    """
    model.eval()
    
    # Compute retain set gradients
    retain_grads = {}
    for batch_idx, (data, target) in enumerate(retain_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Store gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in retain_grads:
                    retain_grads[name] = param.grad.detach().clone()
                else:
                    retain_grads[name] += param.grad.detach().clone()
        
        model.zero_grad()
        if batch_idx >= 5:  # Use smaller subset for efficiency
            break
    
    # Compute forget set gradients  
    forget_grads = {}
    for batch_idx, (data, target) in enumerate(forget_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Store gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in forget_grads:
                    forget_grads[name] = param.grad.detach().clone()
                else:
                    forget_grads[name] += param.grad.detach().clone()
        
        model.zero_grad()
        if batch_idx >= 5:  # Use smaller subset for efficiency
            break
    
    # GA-specific: Negative of retain + positive of forget (CONSERVATIVE)
    ga_gradients = {}
    for name in retain_grads.keys():
        if name in forget_grads:
            # CONSERVATIVE approach: reduce aggression
            retain_component = -0.3 * retain_grads[name]  # Reduced from -1.0
            forget_component = 0.7 * forget_grads[name]   # Emphasize forget less
            ga_gradients[name] = retain_component + forget_component
    
    return ga_gradients

def generate_masks_for_percentage(data_loaders, model, criterion, args, forget_percentage, paths):
    """
    Generate Otsu mask for specific forgetting percentage - UPDATED WITH GA FIXES
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    model.eval()

    # Initialize gradients dictionary
    for name, param in model.named_parameters():
        gradients[name] = 0

    print(f"\n🎯 Generating mask for {forget_percentage*100:.0f}% forgetting")
    print(f"🔧 METHOD: {args.unlearn}")

    # METHOD-SPECIFIC GRADIENT COMPUTATION
    if args.unlearn == "GA":
        if forget_loader is None or len(forget_loader.dataset) == 0:
             print("❌ ERROR: GA method called but forget_loader is empty.")
             return None, 0

        # USE FIXED GA GRADIENT COMPUTATION
        gradients = compute_ga_gradients_fixed(model, retain_loader, forget_loader, 'cuda')
        gradients = scale_ga_gradients(gradients, scale_factor=0.5)  # CONSERVATIVE SCALING

    elif args.unlearn == "FT":
        if retain_loader is None or len(retain_loader.dataset) == 0:
             print("❌ ERROR: FT method called but retain_loader is empty.")
             return None, 0

        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)
            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data
        
    elif args.unlearn == "RL":
        if forget_loader is None or len(forget_loader.dataset) == 0:
             print("❌ ERROR: RL method called but forget_loader is empty.")
             return None, 0
        if retain_loader is None or len(retain_loader.dataset) == 0:
             print("❌ ERROR: RL method called but retain_loader is empty.")
             return None, 0

        forget_gradients = {}
        for name, param in model.named_parameters():
            forget_gradients[name] = 0
            
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)
            loss = -criterion(output_clean, target) 
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        forget_gradients[name] += param.grad.data
            
        retain_gradients = {}
        for name, param in model.named_parameters():
            retain_gradients[name] = 0
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)
            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        retain_gradients[name] += param.grad.data
            
        with torch.no_grad():
            for name in gradients:
                forget_abs = torch.abs_(forget_gradients[name])
                retain_abs = torch.abs_(retain_gradients[name])
                gradients[name] = forget_abs - retain_abs

    # Convert to absolute values for saliency (except RL)
    with torch.no_grad():
        if args.unlearn != "RL":
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

    # FIX for FT/RL Outliers: Clip gradients
    if args.unlearn in ["FT", "RL"]:
        print(f"🔧 Pre-processing {args.unlearn} gradients to handle outliers...")
        
        # Calculate percentile on ABSOLUTE values for both FT and RL
        all_grads_flat = torch.cat([
            g.detach().cpu().float().abs().flatten() for g in gradients.values()
        ])

        if all_grads_flat.numel() > 0:
            # Calculate percentile from the magnitudes
            clip_value = torch.quantile(all_grads_flat, 0.999)
            print(f"    Clipping gradient MAGNITUDES at 99.9th percentile: {clip_value.item():.4f}")

            with torch.no_grad():
                for name in gradients:
                    p99_9 = clip_value.to(gradients[name].device)
                    
                    # Clamp both min and max (magnitude)
                    torch.clamp_(gradients[name], min=-p99_9, max=p99_9)
        else:
            print("    No gradients found to clip.")
    # END FIX

    # DEBUG: Print gradient statistics
    all_grads = debug_gradient_stats(gradients, args.unlearn)

    # 🆕 UPDATED METHOD-AWARE LOGIC WITH GA FIXES
    otsu_method = 'bounded' # Bounded Otsu is the most robust default
    
    if args.unlearn == "GA":
        print("🔧 GA (CONSERVATIVE FIX): Using retention bounds (60-90%)")
        otsu_kwargs = {
            'min_retention': 0.6,  # CHANGED FROM 0.3 to 0.6
            'max_retention': 0.9,  # CHANGED FROM 0.7 to 0.9
            'method': args.unlearn   # Pass method for GA-specific adjustments
        }
        
        # Fallback for near-zero gradients
        if all_grads is not None and np.percentile(all_grads, 99) < 1e-6:
            print("🔧 GA gradients too small, using fixed retention")
            otsu_method = 'fixed'
            otsu_kwargs = {'retention_rate': 0.7}  # Higher retention for GA
    
    elif args.unlearn == "FT":
        print(f"🔧 FT (Balanced): Using retention bounds (50-80%)")
        otsu_kwargs = {
            'min_retention': 0.5, # Keep *at least* 50%
            'max_retention': 0.8, # Keep *at most* 80%
            'method': args.unlearn
        }

    else: # For RL
        print(f"🔧 RL (Conservative): Using retention bounds (60-90%)")
        otsu_kwargs = {
            'min_retention': 0.6, # Keep *at least* 60%
            'max_retention': 0.9, # Keep *at most* 90%
            'method': args.unlearn
        }

    print(f"🔧 Using {otsu_method} Otsu for {args.unlearn} method")

    otsu_function = get_otsu_method(otsu_method, **otsu_kwargs)
    hard_dict = otsu_function(gradients)
    
    if hard_dict is None:
        print("❌ ERROR: Otsu method returned None, using fallback")
        otsu_function = get_otsu_method('fixed', retention_rate=0.7)
        hard_dict = otsu_function(gradients)
    
    # Calculate statistics
    total_params = sum(mask.numel() for mask in hard_dict.values())
    if total_params == 0:
        print("❌ ERROR: Model has no parameters.")
        return None, 0
        
    retained_params = sum(mask.sum().item() for mask in hard_dict.values())
    retention_rate = retained_params / total_params * 100
    
    # RETENTION RATE FALLBACK - UPDATED THRESHOLD
    if retention_rate < 10.0:  # If less than 10% retention (CHANGED FROM 1%)
        print(f"⚠️  Warning: {retention_rate:.1f}% retention - using fallback (70%)")
        fallback_retention = 0.7  # 70% retention (HIGHER FOR SAFETY)
        fallback_otsu_func = get_otsu_method('fixed', retention_rate=fallback_retention)
        hard_dict = fallback_otsu_func(gradients)
        
        retained_params = sum(mask.sum().item() for mask in hard_dict.values())
        retention_rate = retained_params / total_params * 100
        print(f"🔄 Applied fallback: {retention_rate:.1f}% retention")
    
    print(f"📊 Mask Statistics for {forget_percentage*100:.0f}% forgetting:")
    print(f"    Retention rate: {retention_rate:.1f}%")
    print(f"    Method: {args.unlearn}")
    
    return hard_dict, retention_rate

def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
    """
    Mark dataset for specific forgetting percentage
    """
    np.random.seed(seed)
    
    try:
        targets_np = np.array(dataset.targets)
    except:
        targets_np = np.array([s[1] for s in dataset.samples])
        
    total_samples = len(targets_np)
    num_to_forget = int(forget_percentage * total_samples)
    
    if num_to_forget > total_samples:
        num_to_forget = total_samples
        
    forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False)
    
    print(f"🎯 Marking {num_to_forget} samples ({forget_percentage*100:.0f}%) for forgetting")
    
    if hasattr(dataset, 'targets'):
        # Ensure targets is a list for mutation
        if not isinstance(dataset.targets, list):
             dataset.targets = list(dataset.targets)
             
        for idx in forget_indices:
            if dataset.targets[idx] >= 0:
                dataset.targets[idx] = -dataset.targets[idx] - 1
        targets_after = np.array(dataset.targets)
    
    elif hasattr(dataset, 'samples'):
        for idx in forget_indices:
            if dataset.samples[idx][1] >= 0:
                dataset.samples[idx] = (dataset.samples[idx][0], -dataset.samples[idx][1] - 1)
        targets_after = np.array([s[1] for s in dataset.samples])

    else:
        print("❌ ERROR: Unknown dataset structure. Cannot mark targets.")
        return dataset

    forget_count = np.sum(targets_after < 0)
    retain_count = np.sum(targets_after >= 0)
    
    print(f"✅ Marking complete: {forget_count} forget samples, {retain_count} retain samples")
    
    return dataset

def split_marked_dataset(marked_dataset):
    """
    Splits a dataset marked with negative targets into forget and retain datasets.
    """
    forget_dataset = copy.deepcopy(marked_dataset)
    retain_dataset = copy.deepcopy(marked_dataset)
    
    # Case 1: Standard torchvision dataset (e.g., CIFAR10/100)
    if hasattr(marked_dataset, 'data') and hasattr(marked_dataset, 'targets'):
        targets = np.array(marked_dataset.targets)
        forget_mask = targets < 0
        retain_mask = targets >= 0 
        
        # Apply masks to forget dataset
        forget_dataset.data = forget_dataset.data[forget_mask, ...]
        forget_dataset.targets = (-targets[forget_mask] - 1).tolist()
        
        # Apply masks to retain dataset
        retain_dataset.data = retain_dataset.data[retain_mask, ...]
        retain_dataset.targets = targets[retain_mask].tolist()

    # Case 2: ImageFolder-style dataset
    elif hasattr(marked_dataset, 'samples'):
        all_samples = marked_dataset.samples
        forget_samples = []
        retain_samples = []
        
        for (path, target) in all_samples:
            if target < 0: # Forget sample
                forget_samples.append((path, -target - 1))
            else: # Retain sample
                retain_samples.append((path, target))
                
        forget_dataset.samples = forget_samples
        forget_dataset.imgs = forget_samples 
        
        retain_dataset.samples = retain_samples
        retain_dataset.imgs = retain_samples
        
        if hasattr(forget_dataset, 'targets'):
            forget_dataset.targets = [s[1] for s in forget_samples]
        if hasattr(retain_dataset, 'targets'):
            retain_dataset.targets = [s[1] for s in retain_samples]

    else:
        print("❌ ERROR: Unknown dataset structure. Cannot split.")
        return None, None

    return forget_dataset, retain_dataset

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    """
    Creates a new DataLoader for a given dataset, handling empty datasets.
    """
    if dataset is None or len(dataset) == 0:
        return None
    
    utils.setup_seed(seed) # Reset seed for consistent shuffling
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4, 
        pin_memory=True,
        shuffle=shuffle,
    )

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    
    # SETUP ORGANIZED PATHS
    paths = setup_experiment_paths(args)
    paths['organize_by_method'] = getattr(args, 'organize_by_method', True)
    paths['organize_by_percentage'] = getattr(args, 'organize_by_percentage', True)
    
    # Save experiment configuration
    save_experiment_config(args, paths)
    
    # Prepare base dataset
    model, train_loader_full, val_loader, _, _ = utils.setup_model_dataset(args)
    model.cuda()

    # Define forgetting percentages to test
    forget_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    methods = ['GA', 'FT', 'RL']
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model file not found: {args.model_path}")
        return

    print(f"📥 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Model loaded successfully!")

    criterion = nn.CrossEntropyLoss()
    
    # Results storage
    results = {}
    
    # Save progress log
    progress_log_path = os.path.join(paths['logs'], 'mask_generation_progress.log')
    with open(progress_log_path, 'w') as f:
        f.write("Mask Generation Progress Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Started at: {datetime.datetime.now()}\n\n")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"🔄 GENERATING MASKS FOR METHOD: {method}")
        print(f"{'='*60}")
        
        args.unlearn = method
        method_results = {}
        
        for percentage in forget_percentages:
            print(f"\n📊 Processing {percentage*100:.0f}% forgetting...")
            
            original_dataset = copy.deepcopy(train_loader_full.dataset)
            marked_dataset = mark_dataset_for_percentage(original_dataset, percentage, seed=seed)
            
            forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
            
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
            
            # Check if we have required data for the method
            if method == 'GA' and (forget_loader is None or len(forget_dataset) == 0):
                 if percentage == 0.0:
                     print(f"⏭️ Skipping GA for 0% - no forget samples")
                     continue
                 if percentage == 1.0 and (forget_loader is None or len(forget_dataset) == 0):
                     print("❌ ERROR: 100% forgetting but no forget samples found. Check dataset split.")
                     continue
                
            if method in ['FT', 'RL'] and (retain_loader is None or len(retain_dataset) == 0):
                if percentage == 1.0: # Expected case
                    print(f"⏭️ Skipping {method} for 100% - no retain samples")
                    continue
                else:
                    print(f"❌ ERROR: {method} at {percentage*100:.0f}% has no retain samples.")
                    continue

            unlearn_data_loaders = OrderedDict(
                retain=retain_loader, forget=forget_loader, val=val_loader
            )

            # Generate mask
            mask, retention_rate = generate_masks_for_percentage(
                unlearn_data_loaders, model, criterion, args, percentage, paths
            )
            
            if mask is None:
                print(f"❌ Failed to generate mask for {method} at {percentage*100:.0f}%")
                continue
            
            # USE ORGANIZED PATH FOR SAVING
            mask_path = get_mask_path(paths, method, percentage, getattr(args, 'otsu_method', 'conservative'))
            torch.save(mask, mask_path)
            
            method_results[percentage] = {
                'retention_rate': retention_rate,
                'mask_path': mask_path,
                'forget_samples': len(forget_dataset) if forget_dataset else 0,
                'retain_samples': len(retain_dataset) if retain_dataset else 0
            }
            
            print(f"💾 Mask saved to: {mask_path}")
            
            with open(progress_log_path, 'a') as f:
                f.write(f"{method}_{int(percentage*100)}%: {retention_rate:.1f}% retention\n")
        
        results[method] = method_results

    # Save comprehensive results summary
    summary_path = os.path.join(paths['summary'], 'mask_generation_summary.json')
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📈 MASK GENERATION COMPLETE SUMMARY")
    print(f"{'='*60}")
    print(f"📁 All results saved in: {paths['base']}")
    print(f"📊 Summary saved: {summary_path}")
    print(f"📝 Progress log: {progress_log_path}")
    
    for method in methods:
        print(f"\n{method} Method:")
        print("Percentage | Retention | Forget Samples | Retain Samples")
        print("-" * 55)
        for percentage in forget_percentages:
            if percentage in results[method]:
                data = results[method][percentage]
                print(f"{percentage*100:>4.0f}%      | {data['retention_rate']:>8.1f}% | {data['forget_samples']:>13} | {data['retain_samples']:>12}")
            else:
                # Print skipped for 100% FT/RL
                total_samples = len(train_loader_full.dataset)
                forget_samples = int(percentage * total_samples)
                retain_samples = total_samples - forget_samples
                if percentage == 1.0 and method in ['FT', 'RL']:
                    print(f"{percentage*100:>4.0f}%      | {'SKIPPED':>8} | {forget_samples:>13} | {retain_samples:>12}")
                else:
                    print(f"{percentage*100:>4.0f}%      | {'FAILED':>8} | {forget_samples:>13} | {retain_samples:>12}")

if __name__ == "__main__":
    main()