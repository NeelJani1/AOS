import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os
import json
from datetime import datetime
import logging

# Enhanced Otsu thresholding with method-specific retention bounds
def enhanced_otsu_threshold(gradients, method='FT', target_retention_range=(0.5, 0.8)):
    """
    Enhanced Otsu thresholding with method-specific retention bounds and GA fixes
    """
    try:
        # Handle GA method specially - conservative approach
        if method == 'GA':
            # GA gradients are negative and small - use conservative bounds
            target_retention_range = (0.6, 0.9)  # Conservative retention for GA
            
        # Flatten and preprocess gradients
        gradients_flat = torch.cat([g.abs().flatten() for g in gradients.values()])
        
        # Remove zeros and very small values to avoid numerical issues
        gradients_flat = gradients_flat[gradients_flat > 1e-10]
        
        if len(gradients_flat) == 0:
            print("⚠️  All gradients are zero, using default threshold")
            return 0.0
            
        # Log transform for better separation (especially for GA)
        gradients_flat = torch.log(gradients_flat + 1e-10)
        
        # Convert to numpy for Otsu
        grad_np = gradients_flat.cpu().numpy()
        
        # Enhanced Otsu implementation with bounds
        return bounded_otsu(grad_np, target_retention_range, method)
        
    except Exception as e:
        print(f"❌ Error in enhanced_otsu_threshold: {e}")
        # Fallback: return median-based threshold
        gradients_flat = torch.cat([g.abs().flatten() for g in gradients.values()])
        return torch.quantile(gradients_flat, 0.7).item()

def bounded_otsu(data, retention_range=(0.5, 0.8), method='FT'):
    """
    Bounded Otsu thresholding that ensures retention rate stays within specified bounds
    """
    try:
        from skimage.filters import threshold_otsu
        import numpy as np
        
        if len(data) == 0:
            return 0.0
            
        # Calculate Otsu threshold
        thresh = threshold_otsu(data)
        
        # Calculate current retention rate
        retention_rate = np.mean(data >= thresh)
        
        min_retention, max_retention = retention_range
        
        # Adjust threshold to stay within bounds
        if retention_rate < min_retention:
            # Too aggressive - increase retention by lowering threshold
            sorted_vals = np.sort(data)
            target_idx = int(len(sorted_vals) * (1 - min_retention))
            thresh = sorted_vals[target_idx] if target_idx < len(sorted_vals) else sorted_vals[-1]
            print(f"   🔧 {method} Fix: Increased retention to {min_retention*100:.1f}%")
            
        elif retention_rate > max_retention:
            # Too conservative - decrease retention by increasing threshold  
            sorted_vals = np.sort(data)
            target_idx = int(len(sorted_vals) * (1 - max_retention))
            thresh = sorted_vals[target_idx] if target_idx < len(sorted_vals) else sorted_vals[-1]
            print(f"   🔧 {method} Fix: Decreased retention to {max_retention*100:.1f}%")
        
        # Convert back from log space
        final_threshold = np.exp(thresh)
        final_retention = np.mean(data >= thresh)
        
        print(f"🎯 Enhanced Otsu: {final_retention*100:.1f}% retained (target: {min_retention*100:.1f}-{max_retention*100:.1f}%)")
        
        return final_threshold
        
    except Exception as e:
        print(f"❌ Error in bounded_otsu: {e}")
        # Fallback to median
        return np.median(data)

def generate_otsu_mask(gradients, method='FT', percent=10):
    """
    Generate mask using enhanced Otsu thresholding with method-specific handling
    """
    print(f"🔧 METHOD: {method}")
    
    # Method-specific configurations
    method_configs = {
        'GA': {'retention_range': (0.6, 0.9), 'scaling': 0.5, 'description': 'CONSERVATIVE'},
        'FT': {'retention_range': (0.5, 0.8), 'scaling': 1.0, 'description': 'Balanced'}, 
        'RL': {'retention_range': (0.6, 0.9), 'scaling': 1.0, 'description': 'Conservative'}
    }
    
    config = method_configs.get(method, method_configs['FT'])
    
    # Apply method-specific scaling
    if method == 'GA':
        print(f"🔧 Scaling GA gradients by {config['scaling']}x (CONSERVATIVE)")
        gradients = {k: v * config['scaling'] for k, v in gradients.items()}
    
    print(f"🔧 {method} ({config['description']}): Using retention bounds {config['retention_range'][0]*100:.0f}-{config['retention_range'][1]*100:.0f}%")
    
    # Get enhanced Otsu threshold
    threshold = enhanced_otsu_threshold(
        gradients, 
        method=method, 
        target_retention_range=config['retention_range']
    )
    
    print(f"🔧 Using bounded Otsu for {method} method")
    
    # Create mask based on threshold
    mask = {}
    total_params = 0
    retained_params = 0
    
    for name, grad in gradients.items():
        # For GA method, protect critical layers
        if method == 'GA':
            if 'conv1' in name or 'layer1' in name or 'fc' in name:
                mask[name] = torch.ones_like(grad)
                retained_params += mask[name].sum().item()
                total_params += mask[name].numel()
                continue
                
        # Standard masking for other layers/methods
        layer_mask = (grad.abs() >= threshold).float()
        mask[name] = layer_mask
        
        retained_params += layer_mask.sum().item()
        total_params += layer_mask.numel()
    
    retention_rate = retained_params / total_params if total_params > 0 else 0
    
    print(f"📊 Mask Statistics for {percent}% forgetting:")
    print(f"    Retention rate: {retention_rate*100:.1f}%")
    print(f"    Method: {method}")
    
    return mask, retention_rate

def compute_gradients(model, forget_loader, retain_loader, method, device):
    """
    Compute gradients for different unlearning methods
    """
    model.eval()
    gradients = {}
    
    if method == 'GA':
        # Gradient Ascent on forget set
        for batch_idx, (data, target) in enumerate(forget_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = -nn.CrossEntropyLoss()(output, target)  # Negative loss for ascent
            loss.backward()
            
            # Store gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.detach().clone()
                    else:
                        gradients[name] += param.grad.detach().clone()
            break  # Use first batch only for efficiency
            
    elif method == 'FT':
        # Fine-tuning approach - difference between retain and forget gradients
        forget_grads = {}
        retain_grads = {}
        
        # Compute forget gradients
        for batch_idx, (data, target) in enumerate(forget_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in forget_grads:
                        forget_grads[name] = param.grad.detach().clone()
                    else:
                        forget_grads[name] += param.grad.detach().clone()
            break
            
        # Compute retain gradients  
        for batch_idx, (data, target) in enumerate(retain_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in retain_grads:
                        retain_grads[name] = param.grad.detach().clone()
                    else:
                        retain_grads[name] += param.grad.detach().clone()
            break
            
        # Compute difference (forget - retain)
        for name in forget_grads:
            if name in retain_grads:
                gradients[name] = (forget_grads[name] - retain_grads[name]).abs()
            else:
                gradients[name] = forget_grads[name].abs()
                
    elif method == 'RL':
        # Retain Loss minimization
        for batch_idx, (data, target) in enumerate(retain_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.detach().abs().clone()
                    else:
                        gradients[name] += param.grad.detach().abs().clone()
            break
            
    # Clear gradients
    model.zero_grad()
    
    return gradients

def analyze_gradient_statistics(gradients, method):
    """
    Analyze gradient statistics for debugging
    """
    if not gradients:
        print("❌ No gradients computed")
        return
        
    gradients_flat = torch.cat([g.flatten() for g in gradients.values()])
    
    print(f"🔍 DEBUG: {method} Gradient Statistics")
    print(f"   Total values: {len(gradients_flat):,}")
    print(f"   Min: {gradients_flat.min().item():.10f}")
    print(f"   Max: {gradients_flat.max().item():.10f}")
    print(f"   Mean: {gradients_flat.mean().item():.10f}")
    print(f"   Median: {gradients_flat.median().item():.10f}")
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99, 99.9, 99.99]
    percentile_vals = torch.quantile(gradients_flat, torch.tensor([p/100 for p in percentiles]))
    
    print("   Percentiles:")
    for p, val in zip(percentiles, percentile_vals):
        print(f"     {p}%: {val.item():.10f}")
    
    # Count values above different thresholds
    thresholds = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0]
    print("   Values above thresholds:")
    for threshold in thresholds:
        count = (gradients_flat > threshold).sum().item()
        percentage = (count / len(gradients_flat)) * 100
        print(f"     >{threshold:.0e}: {count:,} ({percentage:.4f}%)")

def setup_experiment_paths(args):
    """
    Setup experiment directory structure - RETURNS DICT
    """
    base_dir = args.output_base_dir
    experiment_name = args.experiment_name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{current_time}")

    # Create directory structure
    masks_dir = os.path.join(experiment_dir, 'masks')
    summary_dir = os.path.join(experiment_dir, 'summary') 
    logs_dir = os.path.join(experiment_dir, 'logs')
    unlearning_dir = os.path.join(experiment_dir, 'unlearning')

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(unlearning_dir, exist_ok=True)

    paths = {
        'experiment_dir': experiment_dir,  # This is the main directory path (STRING)
        'masks_dir': masks_dir,
        'summary_dir': summary_dir,
        'logs_dir': logs_dir,
        'unlearning_dir': unlearning_dir
    }

    return paths

def save_experiment_config(args, experiment_dir):
    """
    Save experiment configuration - NOW EXPECTS STRING PATH
    """
    config = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'experiment_name': args.experiment_name,
        'output_base_dir': args.output_base_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Ensure experiment_dir is a string, not a dict
    if isinstance(experiment_dir, dict):
        experiment_dir = experiment_dir['experiment_dir']
    
    config_path = os.path.join(experiment_dir, 'summary', 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Experiment config saved: {config_path}")

def setup_logging(logs_dir):
    """
    Setup logging configuration
    """
    log_path = os.path.join(logs_dir, 'mask_generation_progress.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Method-specific gradient preprocessing
def preprocess_gradients(gradients, method):
    """
    Preprocess gradients based on method requirements
    """
    if method in ['FT', 'RL']:
        # Handle outliers for FT and RL
        print(f"🔧 Pre-processing {method} gradients to handle outliers...")
        
        # Clip extreme values
        all_grads = torch.cat([g.flatten() for g in gradients.values()])
        clip_value = torch.quantile(all_grads, 0.999)  # 99.9th percentile
        print(f"    Clipping gradient MAGNITUDES at 99.9th percentile: {clip_value:.4f}")
        
        gradients = {k: torch.clamp(v, max=clip_value) for k, v in gradients.items()}
        
    return gradients

# Enhanced mask generation with comprehensive error handling
def generate_method_masks(model, forget_loader, retain_loader, device, method, percent, args, paths):
    """
    Generate masks for a specific method with comprehensive error handling
    """
    try:
        print(f"\n🎯 Generating mask for {percent}% forgetting")
        
        # Compute gradients
        gradients = compute_gradients(model, forget_loader, retain_loader, method, device)
        
        if not gradients:
            print(f"❌ No gradients computed for {method}")
            return None, 0
            
        # Analyze gradients for debugging
        analyze_gradient_statistics(gradients, method)
        
        # Preprocess gradients
        gradients = preprocess_gradients(gradients, method)
        
        # Generate Otsu mask
        mask, retention_rate = generate_otsu_mask(gradients, method, percent)
        
        # Save mask
        mask_dir = os.path.join(paths['masks_dir'], method, f"{percent}percent")
        os.makedirs(mask_dir, exist_ok=True)
        
        mask_filename = f"mask_otsu_{method}_{percent}percent_conservative.pt"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        torch.save(mask, mask_path)
        print(f"💾 Mask saved to: {mask_path}")
        
        return mask, retention_rate
        
    except Exception as e:
        print(f"❌ Error generating mask for {method} at {percent}%: {e}")
        return None, 0

# Main mask generation function
def generate_all_masks(model, forget_loaders, retain_loaders, device, args, paths):
    """
    Generate masks for all methods and percentages
    """
    methods = ['GA', 'FT', 'RL']
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"🔄 GENERATING MASKS FOR METHOD: {method}")
        print(f"{'='*60}")
        
        method_results = {}
        
        for percent in percentages:
            print(f"\n📊 Processing {percent}% forgetting...")
            
            # Skip FT/RL for 100% (no retain samples)
            if percent == 100 and method in ['FT', 'RL']:
                print("⏭️ Skipping FT/RL for 100% - no retain samples")
                continue
                
            forget_loader = forget_loaders[percent]
            retain_loader = retain_loaders[percent]
            
            mask, retention_rate = generate_method_masks(
                model, forget_loader, retain_loader, device, method, percent, args, paths
            )
            
            if mask is not None:
                method_results[percent] = {
                    'retention_rate': retention_rate,
                    'forget_samples': len(forget_loader.dataset),
                    'retain_samples': len(retain_loader.dataset)
                }
        
        results[method] = method_results
    
    return results

# Summary generation
def generate_mask_summary(results, paths):
    """
    Generate summary of mask generation results
    """
    summary_path = os.path.join(paths['summary_dir'], 'mask_generation_summary.json')
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📈 MASK GENERATION COMPLETE SUMMARY")
    print(f"{'='*60}")
    print(f"📁 All results saved in: {paths['experiment_dir']}")
    print(f"📊 Summary saved: {summary_path}")
    
    # Print formatted summary
    for method in ['GA', 'FT', 'RL']:
        if method in results:
            print(f"\n{method} Method:")
            print("Percentage | Retention | Forget Samples | Retain Samples")
            print("-" * 55)
            for percent, data in results[method].items():
                print(f"  {percent}%      |    {data['retention_rate']*100:.1f}%  |          {data['forget_samples']} |        {data['retain_samples']}")

# Mask loading utility
def load_mask(mask_path, device='cuda'):
    """
    Load mask from file with error handling
    """
    try:
        if not os.path.exists(mask_path):
            print(f"❌ Mask file not found: {mask_path}")
            return None
            
        mask = torch.load(mask_path, map_location=device)
        print(f"✅ Mask loaded from: {mask_path}")
        return mask
        
    except Exception as e:
        print(f"❌ Error loading mask from {mask_path}: {e}")
        return None
def get_unlearning_dir(paths, method, percentage):
    """
    Helper function to get the specific output directory for an unlearning run.
    """
    percent_int = int(percentage * 100)
    unlearn_dir = os.path.join(paths['unlearning_dir'], method, f"{percent_int}percent")
    os.makedirs(unlearn_dir, exist_ok=True)
    return unlearn_dir

def save_unlearning_results(results_dict, method, percentage, paths):
    """
    Saves the results of a single unlearning run to a JSON file.
    """
    # Get the specific directory for this run
    unlearn_dir = get_unlearning_dir(paths, method, percentage)
    results_path = os.path.join(unlearn_dir, 'unlearning_results.json')
    
    try:
        # Save the results dictionary
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"💾 Unlearning results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error saving unlearning results to {results_path}: {e}")

# Mask application to model
def apply_mask_to_model(model, mask, freeze_masked=True):
    """
    Apply mask to model parameters
    """
    if mask is None:
        print("❌ No mask to apply")
        return
        
    for name, param in model.named_parameters():
        if name in mask:
            # Create masked parameter
            masked_param = param * mask[name]
            param.data.copy_(masked_param)
            
            # Freeze masked parameters if requested
            if freeze_masked:
                param.requires_grad = (mask[name] == 1).any()
    
    print("✅ Mask applied to model parameters")

# Utility to check mask statistics
def get_mask_statistics(mask):
    """
    Get statistics about the mask
    """
    if mask is None:
        return {}
        
    total_params = 0
    retained_params = 0
    
    for name, mask_tensor in mask.items():
        total_params += mask_tensor.numel()
        retained_params += mask_tensor.sum().item()
    
    retention_rate = retained_params / total_params if total_params > 0 else 0
    
    return {
        'total_parameters': total_params,
        'retained_parameters': int(retained_params),
        'retention_rate': retention_rate,
        'pruned_parameters': total_params - int(retained_params)
    }
def find_existing_experiment_dir(experiment_name, output_base_dir="results"):
    """
    Find existing experiment directory with the same name
    """
    if not os.path.exists(output_base_dir):
        return None
        
    # Look for directories that start with the experiment name
    for dir_name in os.listdir(output_base_dir):
        if dir_name.startswith(experiment_name):
            full_path = os.path.join(output_base_dir, dir_name)
            if os.path.isdir(full_path):
                # Check if it has masks directory
                masks_dir = os.path.join(full_path, 'masks')
                if os.path.exists(masks_dir):
                    print(f"🔍 Found existing experiment: {full_path}")
                    return full_path
    return None