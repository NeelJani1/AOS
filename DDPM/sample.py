import torch
import torchvision.utils as tvu
import os
import numpy as np
from models.diffusion import Conditional_Model
from models.ema import EMAHelper
from functions.denoising import generalized_steps_conditional
from datasets import inverse_data_transform
from functions import create_class_labels
import tqdm

def ddim_sample(x, model, c, cond_scale, betas, steps=20, eta=0.0):
    """DDIM sampling - much faster than DDPM"""
    seq = list(np.linspace(0, len(betas)-1, steps).astype(int))
    return generalized_steps_conditional(x, c, seq, model, betas, cond_scale, eta=eta)[0][-1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    import yaml
    from functions import dict2namespace
    with open("configs/cifar10_sample.yml", "r") as f:
        config = yaml.unsafe_load(f)
        config = dict2namespace(config)
    
    # Create model
    model = Conditional_Model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = "results/cifar10/resume_run/ckpts/ckpt.pth"
    states = torch.load(checkpoint_path, map_location=device)
    state_dict = states[0] if isinstance(states, (list, tuple)) else states
    
    # Fix state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Use EMA if available
    if len(states) > 3 and hasattr(config.model, 'ema') and config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[3])
        model = ema_helper.ema_copy(model)
    
    model.eval()
    
    # Generate beta schedule
    def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
        if beta_schedule == "linear":
            return torch.from_numpy(np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)).float().to(device)
        else:
            raise NotImplementedError(beta_schedule)

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    
    # Sampling parameters
    classes, _ = create_class_labels('x0', n_classes=config.data.n_classes)
    n_samples_per_class = 5000
    batch_size = 192  # Larger batch for efficiency
    cond_scale = 2.0
    sampling_steps = 50  # Ultra-fast sampling
    
    sample_dir = "results/cifar10/resume_run/fid_samples_fast"
    os.makedirs(sample_dir, exist_ok=True)
    
    img_id = 0
    
    for class_label in classes:
        print(f"Generating {n_samples_per_class} samples for class {class_label}")
        
        n_rounds = (n_samples_per_class + batch_size - 1) // batch_size
        n_left = n_samples_per_class
        
        with torch.no_grad():
            for j in tqdm.tqdm(range(n_rounds), desc=f"Class {class_label}"):
                n = min(batch_size, n_left)
                
                x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size, device=device)
                c = torch.ones(n, device=device, dtype=int) * class_label
                
                # Fast DDIM sampling
                x = ddim_sample(x, model, c, cond_scale, betas, steps=sampling_steps, eta=0.0)
                x = inverse_data_transform(config, x)
                
                for k in range(n):
                    tvu.save_image(x[k], os.path.join(sample_dir, f"{img_id}.png"), normalize=True)
                    img_id += 1
                
                n_left -= n
                
                # Clear cache every 10 batches
                if j % 10 == 0:
                    torch.cuda.empty_cache()
    
    print(f"Sampling completed! {img_id} samples saved to: {sample_dir}")

if __name__ == "__main__":
    main()