import torch
import torch.nn as nn
from models.diffusion import Conditional_Model 

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def _normalize_param_name(self, name):
        """Remove all wrapper prefixes for consistent naming"""
        name = name.replace('_orig_mod.', '')  # torch.compile prefix
        name = name.replace('module.', '')      # DataParallel prefix
        return name

    def _get_actual_module(self, module):
        """Extract actual module from wrappers"""
        actual_module = module
        
        if isinstance(actual_module, nn.DataParallel):
            actual_module = actual_module.module
            
        if hasattr(actual_module, '_orig_mod'):
            actual_module = actual_module._orig_mod
            
        return actual_module

    def _get_normalized_params(self, module):
        """Get parameters with normalized names"""
        actual_module = self._get_actual_module(module)
        normalized_params = {}
        
        for name, param in actual_module.named_parameters():
            if param.requires_grad:
                normalized_name = self._normalize_param_name(name)
                normalized_params[normalized_name] = param
                
        return normalized_params

    def register(self, module):
        """Register model parameters for EMA tracking"""
        normalized_params = self._get_normalized_params(module)
        
        for norm_name, param in normalized_params.items():
            self.shadow[norm_name] = param.data.clone()
        
        print(f"EMA registered {len(self.shadow)} parameters")

    def update(self, module):
        """Update EMA shadow parameters"""
        normalized_params = self._get_normalized_params(module)

        for norm_name, param in normalized_params.items():
            if norm_name in self.shadow:
                self.shadow[norm_name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[norm_name].data
            else:
                # Auto-register if missing
                self.shadow[norm_name] = param.data.clone()

    def ema(self, module):
        """Apply EMA weights to module"""
        normalized_params = self._get_normalized_params(module)
        
        for norm_name, param in normalized_params.items():
            if norm_name in self.shadow:
                if (param.data is not None and 
                    self.shadow[norm_name].data is not None and
                    param.data.shape == self.shadow[norm_name].data.shape):
                    param.data.copy_(self.shadow[norm_name].data)

    def ema_copy(self, module):
        """Create a copy with EMA weights"""
        device = next(module.parameters()).device 
        actual_module = self._get_actual_module(module)
        
        module_copy = Conditional_Model(actual_module.config).to(device) 
        
        state_dict_to_load = actual_module.state_dict()
        cleaned_state_dict = {}
        for k, v in state_dict_to_load.items():
            k_clean = self._normalize_param_name(k)
            cleaned_state_dict[k_clean] = v

        module_copy.load_state_dict(cleaned_state_dict, strict=True) 
        self.ema(module_copy) 

        if isinstance(module, nn.DataParallel):
            module_copy = nn.DataParallel(module_copy, device_ids=module.device_ids) 

        return module_copy

    def state_dict(self):
        """Return EMA state dict"""
        return self.shadow

    def load_state_dict(self, state_dict):
        """Load EMA state dict with automatic normalization"""
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k_clean = self._normalize_param_name(k)
            cleaned_state_dict[k_clean] = v
        
        self.shadow = cleaned_state_dict
        print(f"EMA loaded {len(self.shadow)} parameters")