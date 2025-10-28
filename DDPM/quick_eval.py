import torch
import torchvision.utils as tvu
import os
from runners.diffusion import Diffusion

def quick_eval():
    """Quick visual evaluation of model quality"""
    
    print("=== QUICK MODEL EVALUATION ===")
    
    # Load model
    class Args:
        config = "cifar10_sample.yml"
        ckpt_folder = "./results/cifar10/resume_run"
        mode = "sample"
        cond_scale = 2.0
        sample_type = "generalized"
        skip_type = "uniform"
        timesteps = 250
        
    class Config:
        data = type('', (), {})()
        data.image_size = 32
        data.channels = 3
        data.n_classes = 10
        
    args = Args()
    config = Config()
    
    diffusion = Diffusion(args, config)
    model = diffusion.load_ema_model()
    
    # Generate samples from each class
    print("Generating samples from all 10 classes...")
    
    os.makedirs("./quick_eval", exist_ok=True)
    
    with torch.no_grad():
        for class_id in range(10):
            x = torch.randn(5, 3, 32, 32, device=diffusion.device)
            classes = torch.full((5,), class_id, device=diffusion.device)
            
            samples = diffusion.sample_image(x, model, classes, args.cond_scale)
            
            # Save samples
            for i in range(5):
                img = (samples[i] + 1) / 2  # Convert to [0,1]
                tvu.save_image(img, f"./quick_eval/class_{class_id}_sample_{i}.png")
    
    print("Samples saved to ./quick_eval/")
    print("\n=== EVALUATION CRITERIA ===")
    print("✅ GOOD: Clear CIFAR-10 objects (cars, animals, etc.)")
    print("❌ BAD:  Blurry noise, artifacts, no recognizable objects")
    print("❌ BAD:  Monochrome or weird colors")

if __name__ == "__main__":
    quick_eval()