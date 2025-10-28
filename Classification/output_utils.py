import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import OrderedDict

def debug_gradient_stats(gradients, method):
    """Debug gradient statistics"""
    print(f"\n🔍 DEBUG: {method} Gradient Statistics")
    
    all_grads = []
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            flattened = grad.abs().flatten().cpu().numpy()
            all_grads.extend(flattened)
    
    if all_grads:
        all_grads = np.array(all_grads)
        print(f"   Total values: {len(all_grads):,}")
        print(f"   Min: {all_grads.min():.10f}")
        print(f"   Max: {all_grads.max():.10f}") 
        print(f"   Mean: {all_grads.mean():.10f}")
        print(f"   Median: {np.median(all_grads):.10f}")
        
        # Check percentiles
        print(f"   Percentiles:")
        for p in [50, 75, 90, 95, 99, 99.9, 99.99]:
            val = np.percentile(all_grads, p)
            print(f"     {p}%: {val:.10f}")
        
        # Count values above different thresholds
        thresholds = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
        print(f"   Values above thresholds:")
        for thresh in thresholds:
            count = np.sum(all_grads > thresh)
            percentage = count / len(all_grads) * 100
            print(f"     >{thresh:.0e}: {count:,} ({percentage:.4f}%)")
    else:
        print("   ❌ No gradients found!")

    return all_grads

def compute_enhanced_otsu_threshold(gradients_flat: torch.Tensor, num_bins: int = 256) -> float:
    """
    Enhanced Otsu threshold computation with better numerical stability
    """
    # Remove outliers for better stability
    gradients_flat = gradients_flat[gradients_flat > 0]  # Only positive values
    if len(gradients_flat) == 0:
        return 0.0
    
    # Use log scale for better distribution handling
    gradients_flat = torch.log(gradients_flat + 1e-10)
    
    min_val, max_val = gradients_flat.min(), gradients_flat.max()
    if min_val == max_val:
        return float(min_val)
    
    # Create histogram
    histogram = torch.histc(gradients_flat, bins=num_bins, min=float(min_val), max=float(max_val))
    bin_edges = torch.linspace(float(min_val), float(max_val), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert to probabilities
    histogram = histogram.float()
    total_pixels = histogram.sum()
    if total_pixels == 0:
        return float(bin_centers[num_bins // 2])
    
    pdf = histogram / total_pixels
    cdf = torch.cumsum(pdf, dim=0)
    
    # Compute between-class variance for all thresholds
    mean_intensity = torch.sum(bin_centers * pdf)
    variance_between = torch.zeros(num_bins, device=gradients_flat.device)
    
    for i in range(1, num_bins):
        if cdf[i] == 0 or cdf[i] == 1:
            continue
            
        mean_low = torch.sum(bin_centers[:i+1] * pdf[:i+1]) / cdf[i]
        mean_high = torch.sum(bin_centers[i+1:] * pdf[i+1:]) / (1 - cdf[i])
        
        variance_between[i] = cdf[i] * (1 - cdf[i]) * (mean_low - mean_high) ** 2
    
    # Find optimal threshold
    if variance_between.max() == 0:
        optimal_idx = num_bins // 2
    else:
        optimal_idx = torch.argmax(variance_between).item()
    
    optimal_threshold = bin_centers[optimal_idx]
    
    # Convert back from log scale
    optimal_threshold = torch.exp(torch.tensor(optimal_threshold)).item()
    
    return optimal_threshold

def bounded_otsu_enhanced(gradients, min_retention=0.3, max_retention=0.7, method='standard'):
    """
    Enhanced Bounded Otsu with method-specific optimizations and GA fixes
    """
    print(f"🔧 Using Enhanced Bounded Otsu (retention: {min_retention*100:.1f}%-{max_retention*100:.1f}%)")
    
    # Flatten all gradients
    all_grads = []
    layer_sizes = {}
    
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            flattened = grad.abs().flatten().cpu()
            all_grads.append(flattened)
            layer_sizes[name] = grad.numel()
    
    if not all_grads:
        print("❌ No gradients found for Otsu thresholding")
        return None, 0, 0
    
    all_grads = torch.cat(all_grads)
    all_grads = all_grads[all_grads > 1e-10]
    
    if len(all_grads) == 0:
        print("❌ All gradients are zero")
        return None, 0, 0
    
    # Use enhanced Otsu threshold
    threshold = compute_enhanced_otsu_threshold(all_grads)
    
    # Create initial mask
    mask = OrderedDict()
    for name, grad in gradients.items():
        if grad is not None:
            mask_tensor = (grad.abs() > threshold).float()
            mask[name] = mask_tensor
    
    # Compute actual retention rate
    total_params = sum(mask[name].numel() for name in mask)
    retained_params = sum(mask[name].sum().item() for name in mask)
    retention_rate = retained_params / total_params
    
    # Apply bounds with adaptive adjustment
    if retention_rate < min_retention:
        # Too aggressive - increase retention by lowering threshold
        target_retention = min_retention
        quantile_threshold = torch.quantile(all_grads, 1 - target_retention).item()
        
        for name, grad in gradients.items():
            mask[name] = (grad.abs() > quantile_threshold).float()
        
        retention_rate = target_retention
        threshold = quantile_threshold
        
    elif retention_rate > max_retention:
        # Too conservative - decrease retention by raising threshold  
        target_retention = max_retention
        quantile_threshold = torch.quantile(all_grads, 1 - target_retention).item()
        
        for name, grad in gradients.items():
            mask[name] = (grad.abs() > quantile_threshold).float()
        
        retention_rate = target_retention
        threshold = quantile_threshold
    
    # METHOD-SPECIFIC ADJUSTMENTS
    if method == 'GA':
        # GA FIX: Ensure we don't mask out critical parameters
        layer_retentions = {}
        for name, layer_mask in mask.items():
            layer_retentions[name] = layer_mask.sum().item() / layer_mask.numel()
        
        # Increase retention for early layers (often more critical)
        for name in mask.keys():
            if 'conv1' in name or 'layer1' in name or 'fc' in name:
                if layer_retentions[name] < 0.7:  # Ensure high retention for critical layers
                    mask[name] = torch.ones_like(mask[name])
                    print(f"   🔧 GA Fix: Increased retention for {name}")
    
    print(f"🎯 Enhanced Otsu: {retention_rate*100:.1f}% retained (target: {min_retention*100:.1f}-{max_retention*100:.1f}%)")
    
    return mask, retention_rate, threshold

def basic_otsu(gradients):
    """Basic Otsu thresholding"""
    print("🔧 Using Basic Otsu thresholding")
    
    # Flatten all gradients
    all_grads = []
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            flattened = grad.abs().flatten().cpu().numpy()
            all_grads.extend(flattened)
    
    if not all_grads:
        print("❌ No gradients found for Otsu thresholding")
        return None
    
    all_grads = np.array(all_grads)
    all_grads = all_grads[all_grads > 1e-10]
    
    if len(all_grads) == 0:
        print("❌ All gradients are zero")
        return None
    
    # Apply Otsu's method
    threshold = otsu_threshold(all_grads)
    
    print(f"📊 Basic Otsu - Threshold: {threshold:.6f}")
    
    # Create mask
    mask = OrderedDict()
    for name, grad in gradients.items():
        if grad is not None:
            mask_tensor = (grad.abs() > threshold).float()
            mask[name] = mask_tensor
    
    return mask

def conservative_otsu(gradients, conservatism=0.3):
    """Conservative Otsu - shifts threshold to retain more parameters"""
    print(f"🔧 Using Conservative Otsu (conservatism: {conservatism})")
    
    # Flatten all gradients
    all_grads = []
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            flattened = grad.abs().flatten().cpu().numpy()
            all_grads.extend(flattened)
    
    if not all_grads:
        print("❌ No gradients found for Otsu thresholding")
        return None
    
    all_grads = np.array(all_grads)
    all_grads = all_grads[all_grads > 1e-10]
    
    if len(all_grads) == 0:
        print("❌ All gradients are zero")
        return None
    
    # Get basic Otsu threshold
    basic_threshold = otsu_threshold(all_grads)
    
    # Apply conservatism: shift threshold toward smaller values
    conservative_threshold = basic_threshold * (1 - conservatism)
    
    print(f"📊 Conservative Otsu - Basic threshold: {basic_threshold:.6f}")
    print(f"   Conservative threshold: {conservative_threshold:.6f}")
    
    # Create mask
    mask = OrderedDict()
    for name, grad in gradients.items():
        if grad is not None:
            mask_tensor = (grad.abs() > conservative_threshold).float()
            mask[name] = mask_tensor
    
    return mask

def bounded_otsu(gradients, min_retention=0.3, max_retention=0.7):
    """Bounded Otsu - ensures retention rate stays within bounds"""
    print(f"🔧 Using Bounded Otsu (retention: {min_retention*100:.1f}%-{max_retention*100:.1f}%)")
    
    # Flatten all gradients
    all_grads = []
    layer_sizes = {}
    
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            flattened = grad.abs().flatten().cpu().numpy()
            all_grads.extend(flattened)
            layer_sizes[name] = grad.numel()
    
    if not all_grads:
        print("❌ No gradients found for Otsu thresholding")
        return None
    
    all_grads = np.array(all_grads)
    all_grads = all_grads[all_grads > 1e-10]
    
    if len(all_grads) == 0:
        print("❌ All gradients are zero")
        return None
    
    # Binary search for threshold that gives desired retention
    low, high = all_grads.min(), all_grads.max()
    target_retention = (min_retention + max_retention) / 2
    
    for _ in range(20):  # Max 20 iterations
        mid = (low + high) / 2
        
        # Calculate retention with current threshold
        retention = calculate_retention(gradients, mid)
        
        if min_retention <= retention <= max_retention:
            break
        elif retention < min_retention:
            high = mid  # Lower threshold to retain more
        else:
            low = mid   # Raise threshold to retain less
    
    final_threshold = mid
    final_retention = calculate_retention(gradients, final_threshold)
    
    print(f"📊 Bounded Otsu - Final threshold: {final_threshold:.6f}")
    print(f"   Final retention: {final_retention*100:.1f}%")
    
    # Create mask
    mask = OrderedDict()
    for name, grad in gradients.items():
        if grad is not None:
            mask_tensor = (grad.abs() > final_threshold).float()
            mask[name] = mask_tensor
    
    return mask

def layer_aware_otsu(gradients, conservatism=0.2):
    """Layer-aware Otsu - applies different thresholds per layer type"""
    print(f"🔧 Using Layer-aware Otsu (conservatism: {conservatism})")
    
    mask = OrderedDict()
    
    for name, grad in gradients.items():
        if grad is not None and grad.numel() > 0:
            layer_grads = grad.abs().flatten().cpu().numpy()
            layer_grads = layer_grads[layer_grads > 1e-10]
            
            if len(layer_grads) == 0:
                # All gradients are zero, retain all parameters
                mask_tensor = torch.ones_like(grad).float()
            else:
                # Layer-specific Otsu
                layer_threshold = otsu_threshold(layer_grads)
                
                # Adjust based on layer type
                if 'conv' in name or 'weight' in name:
                    # Conservative for convolutional layers
                    adjusted_threshold = layer_threshold * (1 - conservatism)
                elif 'bn' in name or 'bias' in name:
                    # More conservative for batch norm and bias
                    adjusted_threshold = layer_threshold * (1 - conservatism * 1.5)
                else:
                    # Standard for other layers
                    adjusted_threshold = layer_threshold
                
                mask_tensor = (grad.abs() > adjusted_threshold).float()
            
            mask[name] = mask_tensor
            
            # Debug info for first few layers
            if len(mask) <= 3:
                retention = mask_tensor.sum().item() / mask_tensor.numel()
                print(f"   {name}: {retention*100:.1f}% retention")
    
    return mask

def fixed_retention_otsu(gradients, retention_rate=0.5):
    """Fixed retention Otsu - uses fixed percentage instead of thresholding"""
    print(f"🔧 Using Fixed Retention Otsu ({retention_rate*100:.0f}% retention)")
    
    mask = OrderedDict()
    for name, grad in gradients.items():
        if grad is not None:
            # Create random mask with fixed retention
            mask_tensor = torch.rand_like(grad) < retention_rate
            mask[name] = mask_tensor.float()
    
    return mask

def otsu_threshold(gradients):
    """Compute Otsu's threshold for a set of gradients"""
    if len(gradients) == 0:
        return 0
    
    # Normalize gradients to [0, 255] for Otsu
    grad_min, grad_max = gradients.min(), gradients.max()
    if grad_max - grad_min < 1e-10:
        return grad_min
    
    normalized = ((gradients - grad_min) / (grad_max - grad_min) * 255).astype(np.uint8)
    
    # Compute histogram
    hist, bin_edges = np.histogram(normalized, bins=256, range=(0, 255))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Otsu's method
    total = len(normalized)
    sum_total = np.dot(bin_centers, hist)
    
    sum_back = 0
    weight_back = 0
    weight_fore = 0
    
    variance_max = 0
    threshold = 0
    
    for i in range(256):
        weight_back += hist[i]
        if weight_back == 0:
            continue
            
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
            
        sum_back += i * hist[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        
        # Calculate between-class variance
        variance_between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        
        if variance_between > variance_max:
            variance_max = variance_between
            threshold = i
    
    # Convert back to original scale
    original_threshold = threshold / 255 * (grad_max - grad_min) + grad_min
    return original_threshold

def calculate_retention(gradients, threshold):
    """Calculate what percentage of parameters would be retained with given threshold"""
    total_params = 0
    retained_params = 0
    
    for name, grad in gradients.items():
        if grad is not None:
            layer_params = grad.numel()
            layer_retained = (grad.abs() > threshold).sum().item()
            
            total_params += layer_params
            retained_params += layer_retained
    
    return retained_params / total_params if total_params > 0 else 0

def get_otsu_method(method_name, **kwargs):
    """Get Otsu method by name - UPDATED WITH ENHANCED BOUNDED OTSU"""
    if method_name == 'basic':
        return basic_otsu
    elif method_name == 'conservative':
        conservatism = kwargs.get('conservatism', 0.3)
        return lambda grads: conservative_otsu(grads, conservatism=conservatism)
    elif method_name == 'bounded':
        min_retention = kwargs.get('min_retention', 0.3)
        max_retention = kwargs.get('max_retention', 0.7)
        method = kwargs.get('method', 'standard')
        return lambda grads: bounded_otsu_enhanced(grads, min_retention=min_retention, max_retention=max_retention, method=method)[0]
    elif method_name == 'layer_aware':
        conservatism = kwargs.get('conservatism', 0.2)
        return lambda grads: layer_aware_otsu(grads, conservatism=conservatism)
    elif method_name == 'fixed':
        retention_rate = kwargs.get('retention_rate', 0.5)
        return lambda grads: fixed_retention_otsu(grads, retention_rate=retention_rate)
    else:
        # Default to conservative
        conservatism = kwargs.get('conservatism', 0.3)
        return lambda grads: conservative_otsu(grads, conservatism=conservatism)

def test_otsu_methods(gradients):
    """Test all Otsu methods and compare their retention rates"""
    print("\n🧪 Testing all Otsu methods...")
    
    methods = {
        'basic': basic_otsu,
        'conservative_0.2': lambda g: conservative_otsu(g, conservatism=0.2),
        'conservative_0.3': lambda g: conservative_otsu(g, conservatism=0.3),
        'conservative_0.4': lambda g: conservative_otsu(g, conservatism=0.4),
        'bounded_30_70': lambda g: bounded_otsu(g, min_retention=0.3, max_retention=0.7),
        'layer_aware': lambda g: layer_aware_otsu(g, conservatism=0.2),
        'fixed_50': lambda g: fixed_retention_otsu(g, retention_rate=0.5),
    }
    
    results = {}
    
    for name, method in methods.items():
        mask = method(gradients)
        if mask is not None:
            total_params = sum(m.numel() for m in mask.values())
            retained_params = sum(m.sum().item() for m in mask.values())
            retention = retained_params / total_params
            results[name] = retention
            print(f"   {name}: {retention*100:.1f}% retention")
    
    return results