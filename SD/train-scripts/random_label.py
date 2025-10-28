import argparse
import os
import sys
import gc
import math

# === FIX: Add parent directory to system path ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ===============================================

# === OPTIMIZATION IMPORTS ===
import torch
from torch.amp import autocast, GradScaler
import bitsandbytes.optim as bnb
# ============================

from dataset import setup_forget_data, setup_model, setup_remain_data
import matplotlib.pyplot as plt
import numpy as np
from convertModels import savemodelDiffusers
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm

# === OPTIMIZATION SETTINGS ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
# =============================

def certain_label(
    class_to_forget,
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # Determine device type (e.g., 'cuda') from device string (e.g., 'cuda:0')
    device_type = device.split(':')[0]
    
    # === MODEL TRAINING SETUP (OPTIMIZED) ===
    print(f"Setting up model on {device}...")
    model = setup_model(config_path, ckpt_path, device)
    
    # === MEMORY FIX: Use bfloat16 to prevent overflow and save memory ===
    model.bfloat16() 

    # === MEMORY FIX: Enable gradient checkpointing ===
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.use_checkpoint = True
        print("Gradient checkpointing enabled.")
    else:
        print("⚠️ Warning: Could not enable gradient checkpointing.")

    criteria = torch.nn.MSELoss()
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    print("Setting up datasets...")
    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)

    # Set only the UNet to train mode
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.train()
        print("UNet set to train mode.")
    else:
        print("⚠️ Warning: Could not set UNet to train mode. Setting entire model to train.")
        model.train()

    # Keep other parts in eval mode
    if hasattr(model, 'first_stage_model'):
        model.first_stage_model.eval()
    if hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None:
        model.cond_stage_model.eval()

    losses = []

    # Choose parameters to train
    parameters = []
    param_names = [] 
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        for name, param in model.model.diffusion_model.named_parameters():
            if train_method == "xattn":
                if "attn2" in name:
                    parameters.append(param)
                    param_names.append(name)
            elif train_method == "full":
                parameters.append(param)
                param_names.append(name)
    else:
        print("⚠️ Error: Could not find model.model.diffusion_model. Cannot select parameters.")
        return

    print(f"Optimizing {len(parameters)} parameter groups.")

    # === MEMORY FIX: Use 8-bit AdamW optimizer ===
    try:
        optimizer = bnb.AdamW8bit(parameters, lr=lr, eps=1e-6)
    except Exception as e:
        print(f"❌ Error creating 8-bit optimizer: {e}")
        return

    # === MEMORY/SPEED FIX: Initialize Gradient Scaler for mixed precision ===
    # === MEMORY/SPEED FIX: Initialize Gradient Scaler for mixed precision ===
# === BF16 FIX: Disable fast path which is not implemented for BFloat16 ===
    

    if mask_path:
        print(f"Loading mask from {mask_path}...")
        mask = torch.load(mask_path, map_location="cpu")
        print(f"Mask loaded. Found {len(mask)} keys.")
        if param_names:
            print(f"   Example param name: {param_names[0]}")
            if param_names[0] in mask:
                 print(f"   Mask key {param_names[0]} found. ✅")
            else:
                 print(f"   ⚠️ Warning: Param name {param_names[0]} NOT FOUND in mask keys.")
                 print(f"   Example mask key: {next(iter(mask.keys()))}")
        
        name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
    else:
        name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"

    # === TRAINING CODE (OPTIMIZED) ===
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        
        # === BUG FIX: Create a cycling iterator for the 'remain' dataset ===
        remain_iter = iter(remain_dl)
        
        # === BUG FIX: Corrected tqdm loop ===
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(enumerate(forget_dl), total=len(forget_dl))
        for i, (forget_images, forget_labels) in pbar:
            
            # === BUG FIX: Get the next 'remain' batch, cycling if necessary ===
            try:
                remain_images, remain_labels = next(remain_iter)
            except StopIteration:
                remain_iter = iter(remain_dl) # Reset iterator
                remain_images, remain_labels = next(remain_iter)

            # Move data to GPU
            forget_images = forget_images.to(device)
            remain_images = remain_images.to(device)

            optimizer.zero_grad(set_to_none=True)

            # === SPEED/MEMORY FIX: Use autocast for mixed precision ===
            with autocast(device_type=device_type, dtype=torch.bfloat16):
                # Prepare prompts
                forget_prompts = [descriptions[label] for label in forget_labels]
                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]
                remain_prompts = [descriptions[label] for label in remain_labels]

                # --- Remain stage ---
                # === SPEED FIX: Removed .permute() ===
                remain_batch = {"jpg": remain_images, "txt": remain_prompts}
                remain_loss = model.shared_step(remain_batch)[0] 

                # --- Forget stage ---
                # === SPEED FIX: Removed .permute() ===
                forget_batch = {"jpg": forget_images, "txt": forget_prompts}
                pseudo_batch = {"jpg": forget_images, "txt": pseudo_prompts}
                
                # Get latents and embeddings
                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                # Sample noise and timesteps
                t = torch.randint(
                    0,
                    model.num_timesteps,
                    (forget_input.shape[0],),
                    device=model.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model.device)

                # Add noise to latents
                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                # Get model predictions
                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                forget_loss = criteria(forget_out, pseudo_out)

                # --- Total loss ---
                loss = forget_loss + alpha * remain_loss

            # === SPEED/MEMORY FIX: Use scaler for backward pass ===
            loss.backward()

            # Apply saliency mask to gradients
            if mask_path:
                with torch.no_grad():
                    # We iterate over `parameters` which we know are in the optimizer
                    for n, p in model.model.diffusion_model.named_parameters():
                        if p.grad is not None:
                            # The mask keys from generate_mask are just the param names
                            if n in mask:
                                p.grad *= mask[n].to(device)

            # === SPEED/MEMORY FIX: Use scaler for optimizer step ===
            optimizer.step()
            
            losses.append(loss.item() / batch_size)
            pbar.set_postfix(loss=loss.item() / batch_size)

    print("Training complete. Saving model...")
    model.eval()
    model.to("cpu")
    gc.collect()

    save_model(
        model,
        name,
        epoch + 1, 
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    print(f"Model saved to models/{name}")

    save_history(losses, name, str(class_to_forget))
    print(f"Loss history saved to models/{name}/loss.png")


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)
    plt.close() # Close plot to free memory


def save_model(
    model,
    name,
    num,
    compvis_config_file=None, # Receive config_path here
    diffusers_config_file=None, # Receive diffusers_config_path here
    device="cpu", # Default device for saving operations
    save_compvis=True,
    save_diffusers=True,
):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"

    # Save CompVis format (state dict) on CPU
    if save_compvis:
        print(f"Saving CompVis state dict to {path}")
        # Ensure model is on CPU before saving state_dict
        model.to("cpu")
        torch.save(model.state_dict(), path)

    # Save Diffusers format
    if save_diffusers:
        print("Saving Model in Diffusers Format...")
        model.to(device)
        # The savemodelDiffusers function expects the *name* to load the .pt file,
        # the config paths, and the device.
        try:
             # === THIS IS THE CORRECTED CALL ===
            savemodelDiffusers(
                name,                   # Arg 1: Name
                compvis_config_file,    # Arg 2: Compvis config path
                diffusers_config_file,  # Arg 3: Diffusers config path
                num,                    # Arg 4: Epoch number
                device                  # Arg 5: Device (Optional, defaults to cpu)
            )
             # ==================================
        except Exception as e:
             print(f"❌ An error occurred during Diffusers saving: {e}")
             print("   Please check the definition and expected arguments of savemodelDiffusers in convertModels.py")


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    # === FIX: Updated default path ===
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/SD/models/ldm/stable-diffusion-v1/.cache/huggingface/download/sd-v1-4.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0", # Changed default
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()

    # --- Argument Validation ---
    if not os.path.exists(args.ckpt_path):
        print(f"❌ Error: Checkpoint path not found: {args.ckpt_path}")
        sys.exit(1)
    if not os.path.exists(args.config_path):
        print(f"❌ Error: Config path not found: {args.config_path}")
        sys.exit(1)
    if args.mask_path and not os.path.exists(args.mask_path):
        print(f"❌ Error: Mask path not found: {args.mask_path}")
        sys.exit(1)
    
    if args.image_size > 256:
         print(f"⚠️ Warning: Image size {args.image_size} may be too large for your GPU.")
         print("   Consider running with --image_size 256 for stability.")

    if not torch.cuda.is_available():
        print("❌ Error: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    try:
        device_idx = int(args.device)
        if device_idx >= torch.cuda.device_count():
            print(f"❌ Error: Invalid GPU device index {device_idx}. Available devices: {torch.cuda.device_count()}")
            sys.exit(1)
        device = f"cuda:{device_idx}"
    except ValueError:
        print(f"❌ Error: Invalid value for --device: {args.device}. Must be an integer index.")
        sys.exit(1)

    classes = int(args.class_to_forget)
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    certain_label(
        classes,
        train_method,
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
    )