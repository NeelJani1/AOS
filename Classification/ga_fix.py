# ga_fix.py
import torch
import torch.nn as nn
from otsu_utils import bounded_otsu_threshold

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
        if batch_idx >= 10:  # Use subset for efficiency
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
        if batch_idx >= 10:  # Use subset for efficiency
            break
    
    # GA-specific: Negative of retain + positive of forget (conservative)
    ga_gradients = {}
    for name in retain_grads.keys():
        if name in forget_grads:
            # Conservative approach: reduce aggression
            retain_component = -0.3 * retain_grads[name]  # Reduced from -1.0
            forget_component = 0.7 * forget_grads[name]   # Emphasize forget less
            ga_gradients[name] = retain_component + forget_component
    
    return ga_gradients

def create_ga_mask_fixed(gradients, method='GA'):
    """
    Fixed GA mask creation with conservative Otsu thresholding
    """
    # GA-specific: Use absolute values with conservative scaling
    gradients_abs = {k: torch.abs(v) * 0.5 for k, v in gradients.items()}  # Reduced scaling
    
    # Much more conservative retention range for GA
    retention_range = (0.6, 0.9)  # Changed from (0.3, 0.7)
    
    # Apply bounded Otsu with conservative settings
    mask, retention_rate, threshold = bounded_otsu_threshold(
        gradients_abs, 
        retention_range=retention_range,
        num_bins=128  # More bins for finer control
    )
    
    print(f"🎯 GA Adaptive Otsu (FIXED): {retention_rate*100:.1f}% parameters retained")
    return mask, retention_rate