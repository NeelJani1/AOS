import torch
import os
import sys
import gc
import argparse
from tqdm import tqdm
import math

# === NEW IMPORTS for AMP and 8-bit Optimizer ===
from torch.amp import autocast, GradScaler
import bitsandbytes.optim as bnb
# ===============================================

# ==================== MEMORY OPTIMIZATION SETTINGS ====================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Clear GPU cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# ==================== MONKEY PATCH REMOVED ====================
print("🔧 (Monkey patches removed to allow fp16 model)")
# ==============================================================

# ==================== IMPORT AFTER PATCHES ====================
# Ensure the path is correct relative to where generate_mask.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dataset import setup_forget_data, setup_forget_nsfw_data, setup_model
# Import the modified util file containing the fixed GroupNorm32
import ldm.modules.diffusionmodules.util # Make sure the fix is applied here

def generate_mask(
    classes,
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=256,
    num_timesteps=1000,
):
    """
    Generate mask for concept erasure with memory optimizations
    """
    # === FIX: Enable anomaly detection to find NaNs ===
    torch.autograd.set_detect_anomaly(True)
    
    print(f"🚀 Starting mask generation with batch_size={batch_size}, image_size={image_size}")

    # Memory optimization at start
    torch.cuda.empty_cache()
    gc.collect()

    # ==================== MODEL SETUP ====================
    print("📦 Setting up model...")
    model = setup_model(config_path, ckpt_path, device)

    # === FIX: Convert model to bfloat16 to prevent fp16 overflow ===
    model.bfloat16()

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.use_checkpoint = True
    else:
        print("⚠️ Warning: Could not find model.model.diffusion_model to enable checkpointing.")


    # Set model to train mode for the diffusion model parameters only
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.train()  # This enables gradients
    else:
        print("⚠️ Warning: Could not set model.model.diffusion_model to train mode.")
        model.train() # Fallback to setting the whole model to train

    # Keep the rest of the model in eval mode
    if hasattr(model, 'first_stage_model'):
        model.first_stage_model.eval()
    if hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None:
        model.cond_stage_model.eval()

    try:
        train_dl, descriptions = setup_forget_data(classes, batch_size, image_size)
    except Exception as e:
        print(f"❌ Error setting up forget data: {e}")
        print("Please ensure 'frgfm/imagenette' dataset is accessible or downloaded.")
        return


    # Verify model is in bfloat16
    try:
        model_dtype = next(model.parameters()).dtype
        print(f"✅ Model dtype: {model_dtype}")
    except StopIteration:
        print("⚠️ Warning: Model has no parameters.")
        model_dtype = torch.bfloat16 # Assume bf16 if model converted

    # ==================== TRAINING SETUP ====================
    criteria = torch.nn.MSELoss()

    # --- FIX: Added eps=1e-6 to prevent nan gradients with 8-bit optimizer ---
    new_eps = 1e-6 
    
    # Use 8-bit AdamW to save optimizer state memory
    try:
        # Check if the diffusion model path exists before creating optimizer
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            optimizer = bnb.AdamW8bit(
                model.model.diffusion_model.parameters(), 
                lr=lr, 
                eps=new_eps # <-- FIX for nan grads
            )
        else:
            print("⚠️ Warning: model.model.diffusion_model not found. Optimizing all model parameters.")
            optimizer = bnb.AdamW8bit(
                model.parameters(), 
                lr=lr, 
                eps=new_eps # <-- FIX for nan grads
            )
    except Exception as e:
        print(f"❌ Error creating 8-bit optimizer: {e}")
        print("Ensure 'bitsandbytes' is installed correctly.")
        return

    # Update GradScaler API
    scaler = GradScaler('cuda')

    gradients = {}
    try:
        # Check if the diffusion model path exists before accessing named_parameters
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            target_params = model.model.diffusion_model.named_parameters()
        else:
            print("⚠️ Warning: model.model.diffusion_model not found. Accumulating gradients for all parameters.")
            target_params = model.named_parameters()

        for name, param in target_params:
            # Check if param requires grad before initializing gradient dict entry
            if param.requires_grad:
                gradients[name] = 0
            else:
                print(f"   Skipping {name} (requires_grad=False)")
    except Exception as e:
        print(f"❌ Error initializing gradient dictionary: {e}")
        return


    # ==================== TRAINING LOOP WITH MEMORY OPTIMIZATION ====================
    print(f"🎯 Starting training with {len(train_dl)} batches...")

    # Determine device type (e.g., 'cuda') from device string (e.g., 'cuda:0')
    dev_type = device.split(':')[0]

    with tqdm(total=len(train_dl), desc="Generating Mask") as t:
        for i, batch_data in enumerate(train_dl):
            # Handle potential dataset structure issues
            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    images = batch_data
                    labels = torch.zeros(images.shape[0], dtype=torch.long)
                    print(f"⚠️ Warning: Unexpected batch format at index {i}. Assuming image-only batch.")
            except Exception as e:
                print(f"❌ Error processing batch {i}: {e}")
                continue # Skip this batch


            # Memory optimization: clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            optimizer.zero_grad(set_to_none=True)  # More memory efficient

            images = images.to(device)
            try:
                if isinstance(labels, torch.Tensor):
                    labels_list = labels.tolist()
                else:
                    labels_list = list(labels)
                null_prompts = ["" for _ in labels_list]
                prompts = [descriptions[label_idx] for label_idx in labels_list]
            except IndexError:
                print(f"❌ Error: Label index out of range for descriptions at batch {i}.")
                continue # Skip batch if labels don't match descriptions
            except Exception as e:
                print(f"❌ Error preparing prompts for batch {i}: {e}")
                continue # Skip batch on other prompt errors


            # Prepare batches
            try:
                if images.dim() == 4 and images.shape[1] == 3: # Likely (B, C, H, W)
                    forget_batch = {"jpg": images, "txt": prompts}
                    null_batch = {"jpg": images, "txt": null_prompts}
                else:
                    print(f"❌ Error: Unexpected image tensor shape {images.shape} at batch {i}.")
                    continue
            except Exception as e:
                print(f"❌ Error preparing batch dicts at batch {i}: {e}")
                continue


            # ==================== MEMORY-EFFICIENT ENCODING ====================
            try:
                with torch.no_grad():  # Save memory during encoding
                    first_stage_key = getattr(model, 'first_stage_key', 'jpg')
                    cond_stage_key = getattr(model, 'cond_stage_key', 'txt')

                    forget_input, forget_emb = model.get_input(
                        forget_batch, first_stage_key, cond_key=cond_stage_key
                    )
                    null_input, null_emb = model.get_input(
                        null_batch, first_stage_key, cond_key=cond_stage_key
                    )
            except AttributeError as e:
                print(f"❌ Error: Model missing expected attribute during get_input (likely VAE/CLIP keys): {e}")
                return # Stop if essential model attributes are missing
            except Exception as e:
                print(f"❌ Error during model input encoding at batch {i}: {e}")
                continue # Skip batch on encoding errors


            # ==================== DIFFUSION & LOSS (WITH AMP) ====================
            try:
                # === FIX: Use bfloat16 for autocast ===
                with autocast(dev_type, dtype=torch.bfloat16):
                    t_val = torch.randint(
                        0, model.num_timesteps, (forget_input.shape[0],), device=device
                    ).long()
                    noise = torch.randn_like(forget_input, device=device)

                    forget_noisy = model.q_sample(x_start=forget_input, t=t_val, noise=noise)

                    forget_out = model.apply_model(forget_noisy, t_val, forget_emb)
                    null_out = model.apply_model(forget_noisy, t_val, null_emb)

                    # ==================== LOSS CALCULATION ====================
                    preds = (1 + c_guidance) * forget_out - c_guidance * null_out
                    loss = -criteria(noise, preds)

            except AttributeError as e:
                print(f"❌ Error: Model missing expected attribute during diffusion step (q_sample/apply_model): {e}")
                return # Stop if essential model methods are missing
            except RuntimeError as e:
                if "shape" in str(e) or "size" in str(e):
                    print(f"❌ Possible Tensor Shape Mismatch Error at batch {i}: {e}")
                else:
                    # This will now catch the NaN error if anomaly detection is on
                    print(f"❌ RuntimeError during diffusion/loss forward pass at batch {i}: {e}")
                continue # Skip batch on forward pass errors
            except Exception as e:
                print(f"❌ Error during diffusion/loss forward pass at batch {i}: {e}")
                continue # Skip batch


            # ==================== BACKWARD PASS (WITH SCALER) ====================
            try:
                scaler.scale(loss).backward()

                # === FIX: Comment out scaler.unscale_ ===
                # scaler.unscale_(optimizer)

                if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                    target_params_for_clip = model.model.diffusion_model.parameters()
                else:
                    print("⚠️ Warning: model.model.diffusion_model not found for clipping. Clipping all parameters.")
                    target_params_for_clip = model.parameters()

                torch.nn.utils.clip_grad_norm_(target_params_for_clip, max_norm=1.0)
            except RuntimeError as e:
                if "gradient" in str(e).lower() and "non-finite" in str(e).lower():
                    print(f"📈 Detected non-finite gradients (NaN/Inf) at batch {i}. Loss: {loss.item()}. Skipping gradient accumulation.")
                elif "memory" in str(e).lower():
                    print(f"❌ CUDA Out of Memory during backward pass/clipping at batch {i}: {e}")
                else:
                    print(f"❌ RuntimeError during backward pass or clipping at batch {i}: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue # Skip gradient accumulation for this batch
            except Exception as e:
                print(f"❌ Error during backward pass or clipping at batch {i}: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue # Skip gradient accumulation


            # ==================== GRADIENT ACCUMULATION ====================
            try:
                with torch.no_grad():
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        target_params_for_accum = model.model.diffusion_model.named_parameters()
                    else:
                        target_params_for_accum = model.named_parameters()

                    for name, param in target_params_for_accum:
                        if param.requires_grad and param.grad is not None:
                            if name in gradients:
                                gradients[name] += param.grad.data.cpu()
                            else:
                                print(f"⚠️ Warning: Gradient for {name} found but not initialized in dict. Skipping accumulation.")
                        elif param.requires_grad and param.grad is None:
                            print(f"⚠️ Warning: Grad is None for parameter {name} at batch {i}. This might indicate it wasn't used in loss computation.")
            except Exception as e:
                print(f"❌ Error during gradient accumulation at batch {i}: {e}")
                continue


            t.update(1)
            loss_item = loss.item()
            if not math.isfinite(loss_item):
                print(f"\n📉 Loss is non-finite ({loss_item}) at batch {i}. Check gradients/inputs.")
            t.set_postfix(loss=f"{loss_item:.4f}")

    # ==================== MASK GENERATION (RAM EFFICIENT) ====================
    print(f"🎭 Generating final mask (RAM efficient)...")
    try:
        with torch.no_grad():
            if not gradients:
                print("❌ Error: No gradients were accumulated. Cannot generate mask.")
                return

            # --- START: SYSTEM RAM CRASH FIX ---
            # Replaced the argsort-based method with a kthvalue-based method
            # This is significantly more RAM-efficient and avoids crashing WSL
            
            print("   Calculating absolute gradients...")
            for name in list(gradients.keys()): # Iterate over keys list for safe deletion
                if isinstance(gradients[name], torch.Tensor):
                    gradients[name] = torch.abs_(gradients[name])
                elif gradients[name] == 0:
                    del gradients[name]
                else:
                    print(f"⚠️ Warning: Unexpected type in gradients dict for {name}: {type(gradients[name])}")
                    del gradients[name] # Remove unexpected entries
            
            mask_path = os.path.join("mask", str(classes))
            os.makedirs(mask_path, exist_ok=True)
            
            threshold_list = [0.5] # We want to keep the top 50%
            for threshold in threshold_list:
                
                # Filter out non-tensor or zero entries
                valid_gradients_list = [
                    tensor.flatten() for tensor in gradients.values() 
                    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
                ]

                if not valid_gradients_list:
                    print("❌ Error: No valid gradients found after processing. Cannot generate mask.")
                    return

                # Concatenate all absolute gradient values (this still uses RAM, but temporarily)
                try:
                    all_abs_elements = torch.cat(valid_gradients_list)
                except Exception as e:
                    print(f"❌ Error during torch.cat, likely system OOM: {e}")
                    return

                total_params = all_abs_elements.numel()
                print(f"   Total parameters with gradients: {total_params}")

                k = int(total_params * threshold)
                k = max(1, min(k, total_params)) # Ensure k is a valid index
                print(f"   Finding threshold value (k={k} smallest)...")
                
                # Use torch.kthvalue - EXTREMELY RAM EFFICIENT
                threshold_val = torch.kthvalue(all_abs_elements, k).values
                
                if torch.isnan(threshold_val) or torch.isinf(threshold_val):
                    print(f"❌ Error: Calculated threshold is {threshold_val.item()}. Gradients contain NaNs/Infs.")
                    threshold_val = torch.tensor(float('inf')) # Set to infinity to create empty mask
                
                print(f"   Threshold gradient magnitude: {threshold_val.item()}")

                # Free the giant tensor from RAM immediately
                del all_abs_elements
                del valid_gradients_list
                gc.collect()

                print("   Generating mask layer by layer...")
                hard_dict = {}
                total_kept = 0
                for key, tensor in gradients.items():
                    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        continue
                    
                    mask = (tensor >= threshold_val).to(torch.uint8)
                    hard_dict[key] = mask
                    total_kept += mask.sum().item()

                percent_kept = (total_kept / total_params * 100) if total_params > 0 else 0
                print(f"   Mask generated. Kept {total_kept} parameters ({percent_kept:.2f}%)")

                mask_file = os.path.join(mask_path, f"mask_threshold_{threshold:.2f}.pt")
                torch.save(hard_dict, mask_file)
                print(f"💾 Mask saved: {mask_file}")
            
            # --- END: SYSTEM RAM CRASH FIX ---

    except Exception as e:
        print(f"❌ Error during final mask generation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return

    print("✅ Mask generation completed!")


def generate_nsfw_mask(
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=256,
    num_timesteps=1000,
):
    """
    Generate mask for NSFW content erasure
    """
    # === FIX: Enable anomaly detection to find NaNs ===
    torch.autograd.set_detect_anomaly(True)

    print(f"🚀 Starting NSFW mask generation with batch_size={batch_size}")

    # Memory optimization
    torch.cuda.empty_cache()
    gc.collect()

    # Model setup
    model = setup_model(config_path, ckpt_path, device)

    # === FIX: Convert model to bfloat16 to prevent fp16 overflow ===
    model.bfloat16()

    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.use_checkpoint = True
    else:
        print("⚠️ Warning: Could not find model.model.diffusion_model to enable checkpointing (NSFW).")


    # Set diffusion model to train mode for gradients
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        model.model.diffusion_model.train()
    else:
        print("⚠️ Warning: Could not set model.model.diffusion_model to train mode (NSFW).")
        model.train() # Fallback

    if hasattr(model, 'first_stage_model'):
        model.first_stage_model.eval()
    if hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None:
        model.cond_stage_model.eval()

    try:
        forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)
    except Exception as e:
        print(f"❌ Error setting up NSFW data: {e}")
        print("Please ensure 'data/nsfw' and 'data/not-nsfw' datasets exist locally.")
        return


    # Verify model dtype
    try:
        model_dtype = next(model.parameters()).dtype
        print(f"✅ Model dtype: {model_dtype}")
    except StopIteration:
        print("⚠️ Warning: Model has no parameters (NSFW).")
        model_dtype = torch.bfloat16

    criteria = torch.nn.MSELoss()

    # --- FIX: Added eps=1e-6 to prevent nan gradients with 8-bit optimizer ---
    new_eps = 1e-6

    # Use 8-bit AdamW
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            optimizer = bnb.AdamW8bit(
                model.model.diffusion_model.parameters(), 
                lr=lr,
                eps=new_eps # <-- FIX for nan grads
            )
        else:
            print("⚠️ Warning: model.model.diffusion_model not found. Optimizing all parameters (NSFW).")
            optimizer = bnb.AdamW8bit(
                model.parameters(), 
                lr=lr,
                eps=new_eps # <-- FIX for nan grads
            )
    except Exception as e:
        print(f"❌ Error creating 8-bit optimizer (NSFW): {e}")
        return

    # Update GradScaler API
    scaler = GradScaler('cuda')

    word_nude = "a photo of a nude person"

    gradients = {}
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            target_params_nsfw = model.model.diffusion_model.named_parameters()
        else:
            print("⚠️ Warning: model.model.diffusion_model not found. Accumulating gradients for all parameters (NSFW).")
            target_params_nsfw = model.named_parameters()

        for name, param in target_params_nsfw:
            if param.requires_grad:
                gradients[name] = 0
            else:
                print(f"   Skipping {name} (requires_grad=False) (NSFW)")
    except Exception as e:
        print(f"❌ Error initializing gradient dictionary (NSFW): {e}")
        return

    # Determine device type
    dev_type = device.split(':')[0]

    # Training loop
    with tqdm(total=len(forget_dl), desc="NSFW Mask") as t:
        for i, images in enumerate(forget_dl):
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)

            current_batch_size = images.shape[0]
            null_prompts = [""] * current_batch_size
            prompts = [word_nude] * current_batch_size

            # Prepare batches
            try:
                if images.dim() == 4 and images.shape[1] == 3: # Likely (B, C, H, W)
                    forget_batch = {"jpg": images, "txt": prompts}
                    null_batch = {"jpg": images, "txt": null_prompts}
                else:
                    print(f"❌ Error: Unexpected image tensor shape {images.shape} at batch {i} (NSFW).")
                    continue
            except Exception as e:
                print(f"❌ Error preparing batch dicts at batch {i} (NSFW): {e}")
                continue

            try:
                # Encoding (no gradients needed here)
                with torch.no_grad():
                    first_stage_key = getattr(model, 'first_stage_key', 'jpg')
                    cond_stage_key = getattr(model, 'cond_stage_key', 'txt')
                    forget_input, forget_emb = model.get_input(forget_batch, first_stage_key, cond_key=cond_stage_key)
                    null_input, null_emb = model.get_input(null_batch, first_stage_key, cond_key=cond_stage_key)

                # Diffusion Forward & Loss (needs gradients for U-Net)
                # === FIX: Use bfloat16 for autocast ===
                with autocast(dev_type, dtype=torch.bfloat16):
                    t_val = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
                    noise = torch.randn_like(forget_input, device=device)
                    forget_noisy = model.q_sample(x_start=forget_input, t=t_val, noise=noise)

                    forget_out = model.apply_model(forget_noisy, t_val, forget_emb)
                    null_out = model.apply_model(forget_noisy, t_val, null_emb)

                    preds = (1 + c_guidance) * forget_out - c_guidance * null_out
                    loss = -criteria(noise, preds)

            except AttributeError as e:
                print(f"❌ Error: Model missing expected attribute during forward pass (NSFW): {e}")
                return
            except RuntimeError as e:
                if "shape" in str(e) or "size" in str(e):
                    print(f"❌ Possible Tensor Shape Mismatch Error at batch {i} (NSFW): {e}")
                else:
                    print(f"❌ RuntimeError during forward pass at batch {i} (NSFW): {e}")
                continue
            except Exception as e:
                print(f"❌ Error during forward pass at batch {i} (NSFW): {e}")
                continue


            # Backward pass
            try:
                scaler.scale(loss).backward()

                # === FIX: Comment out scaler.unscale_ ===
                # scaler.unscale_(optimizer)

                if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                    target_params_for_clip_nsfw = model.model.diffusion_model.parameters()
                else:
                    print("⚠️ Warning: model.model.diffusion_model not found for clipping (NSFW). Clipping all parameters.")
                    target_params_for_clip_nsfw = model.parameters()

                torch.nn.utils.clip_grad_norm_(target_params_for_clip_nsfw, max_norm=1.0)
            except RuntimeError as e:
                if "gradient" in str(e).lower() and "non-finite" in str(e).lower():
                    print(f"📈 Detected non-finite gradients (NaN/Inf) at batch {i} (NSFW). Loss: {loss.item()}. Skipping grad accum.")
                elif "memory" in str(e).lower():
                    print(f"❌ CUDA Out of Memory during backward/clipping at batch {i} (NSFW): {e}")
                else:
                    print(f"❌ RuntimeError during backward pass or clipping at batch {i} (NSFW): {e}")
                optimizer.zero_grad(set_to_none=True)
                continue
            except Exception as e:
                print(f"❌ Error during backward pass or clipping at batch {i} (NSFW): {e}")
                optimizer.zero_grad(set_to_none=True)
                continue


            # Accumulate scaled gradients
            try:
                with torch.no_grad():
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        target_params_for_accum_nsfw = model.model.diffusion_model.named_parameters()
                    else:
                        target_params_for_accum_nsfw = model.named_parameters()

                    for name, param in target_params_for_accum_nsfw:
                        if param.requires_grad and param.grad is not None:
                            if name in gradients:
                                gradients[name] += param.grad.data.cpu()
                            else:
                                print(f"⚠️ Warning: Grad for {name} found but not initialized (NSFW). Skipping.")
                        elif param.requires_grad and param.grad is None:
                            print(f"⚠️ Warning: Grad is None for {name} at batch {i} (NSFW).")
            except Exception as e:
                print(f"❌ Error during gradient accumulation at batch {i} (NSFW): {e}")
                continue


            t.update(1)
            loss_item_nsfw = loss.item()
            if not math.isfinite(loss_item_nsfw):
                print(f"\n📉 Loss is non-finite ({loss_item_nsfw}) at batch {i} (NSFW).")
            t.set_postfix(loss=f"{loss_item_nsfw:.4f}")

    # ==================== MASK GENERATION (RAM EFFICIENT) ====================
    print(f"🎭 Generating final mask (RAM efficient)...")
    try:
        with torch.no_grad():
            if not gradients:
                print("❌ Error: No gradients accumulated (NSFW). Cannot generate mask.")
                return

            # --- START: SYSTEM RAM CRASH FIX ---
            # Using the same RAM-efficient method as in generate_mask
            
            print("   Calculating absolute gradients...")
            for name in list(gradients.keys()): # Iterate over keys list for safe deletion
                if isinstance(gradients[name], torch.Tensor):
                    gradients[name] = torch.abs_(gradients[name])
                elif gradients[name] == 0:
                    del gradients[name]
                else:
                    print(f"⚠️ Warning: Unexpected type in gradients dict for {name}: {type(gradients[name])}")
                    del gradients[name] # Remove unexpected entries
            
            os.makedirs("mask", exist_ok=True)
            
            threshold_list = [0.5] # We want to keep the top 50%
            for threshold in threshold_list:
                
                valid_gradients_list = [
                    tensor.flatten() for tensor in gradients.values() 
                    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
                ]

                if not valid_gradients_list:
                    print("❌ Error: No valid gradients found after processing (NSFW). Cannot generate mask.")
                    return

                try:
                    all_abs_elements = torch.cat(valid_gradients_list)
                except Exception as e:
                    print(f"❌ Error during torch.cat, likely system OOM (NSFW): {e}")
                    return

                total_params = all_abs_elements.numel()
                print(f"   Total parameters with gradients: {total_params}")

                k = int(total_params * threshold)
                k = max(1, min(k, total_params)) # Ensure k is a valid index
                print(f"   Finding threshold value (k={k} smallest)...")
                
                threshold_val = torch.kthvalue(all_abs_elements, k).values
                
                if torch.isnan(threshold_val) or torch.isinf(threshold_val):
                    print(f"❌ Error: Calculated threshold is {threshold_val.item()} (NSFW).")
                    threshold_val = torch.tensor(float('inf'))
                
                print(f"   Threshold gradient magnitude: {threshold_val.item()}")

                # Free the giant tensor from RAM immediately
                del all_abs_elements
                del valid_gradients_list
                gc.collect()

                print("   Generating mask layer by layer...")
                hard_dict = {}
                total_kept = 0
                for key, tensor in gradients.items():
                    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        continue
                    
                    mask = (tensor >= threshold_val).to(torch.uint8)
                    hard_dict[key] = mask
                    total_kept += mask.sum().item()

                percent_kept = (total_kept / total_params * 100) if total_params > 0 else 0
                print(f"   Mask generated. Kept {total_kept} parameters ({percent_kept:.2f}%)")

                mask_file = f"mask/nude_threshold_{threshold:.2f}.pt"
                torch.save(hard_dict, mask_file)
                print(f"💾 NSFW mask saved: {mask_file}")
            
            # --- END: SYSTEM RAM CRASH FIX ---

    except Exception as e:
        print(f"❌ Error during final NSFW mask generation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("✅ NSFW mask generation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks for concept erasure")

    # Training parameters
    parser.add_argument("--classes", type=str, default="0", help="Class to erase (Imagenette index)")
    parser.add_argument("--c_guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (MUST BE 1 for mask generation)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (typically 1 for mask generation)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (used for optimizer init, but no steps are taken)")

    # Model paths
    default_ckpt_path = os.path.join(parent_dir, 'models', 'ldm', 'stable-diffusion-v1', '.cache', 'huggingface', 'download', 'sd-v1-4.ckpt')
    default_config_path = os.path.join(parent_dir, 'configs', 'stable-diffusion', 'v1-inference.yaml')
    default_diffusers_config_path = os.path.join(parent_dir, 'diffusers_unet_config.json')

    parser.add_argument("--ckpt_path", type=str,
                        default=default_ckpt_path,
                        help="Path to Stable Diffusion checkpoint (.ckpt)")
    parser.add_argument("--config_path", type=str,
                        default=default_config_path,
                        help="Path to model config file (v1-inference.yaml)")
    parser.add_argument("--diffusers_config_path", type=str,
                        default=default_diffusers_config_path,
                        help="Path to diffusers UNet config (if needed, currently unused in script logic)")

    # Hardware settings
    parser.add_argument("--device", type=str, default="0", help="GPU device index (e.g., '0')")
    parser.add_argument("--image_size", type=int, default=256, help="Image size to process (e.g., 256 or 512)")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps model was trained with")
    parser.add_argument("--nsfw", action="store_true", help="Generate mask for NSFW concept instead of Imagenette class")

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.ckpt_path):
        print(f"❌ Error: Checkpoint path not found: {args.ckpt_path}")
        sys.exit(1)
    if not os.path.exists(args.config_path):
        print(f"❌ Error: Config path not found: {args.config_path}")
        sys.exit(1)
    if args.batch_size != 1:
        print("⚠️ Warning: Batch size is set to > 1. Mask generation typically requires batch_size=1. Forcing batch_size=1.")
        args.batch_size = 1


    # Parse arguments
    try:
        classes = int(args.classes) if not args.nsfw else -1
    except ValueError:
        print(f"❌ Error: Invalid value for --classes: {args.classes}. Must be an integer index.")
        sys.exit(1)

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


    print("=" * 60)
    print("🎯 MASK GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🖼️  Image size: {args.image_size}")
    print(f"⚡ Device: {device}")
    if args.nsfw:
        print(f"🎨 Target: NSFW Concept")
    else:
        print(f"🎨 Target Class (Imagenette): {classes}")
    print(f"🧠 NSFW mode active: {args.nsfw}")
    print(f"🔢 Learning Rate (placeholder): {args.lr}")
    print(f"⏳ Timesteps: {args.num_timesteps}")
    print(f"🔧 Config Path: {args.config_path}")
    print(f"💾 Ckpt Path: {args.ckpt_path}")
    print("=" * 60)


    # --- Execute ---
    try:
        if args.nsfw:
            generate_nsfw_mask(
                args.c_guidance, args.batch_size, args.epochs, args.lr,
                args.config_path, args.ckpt_path, args.diffusers_config_path,
                device, args.image_size, args.num_timesteps
            )
        else:
            generate_mask(
                classes, args.c_guidance, args.batch_size, args.epochs, args.lr,
                args.config_path, args.ckpt_path, args.diffusers_config_path,
                device, args.image_size, args.num_timesteps
            )
    except Exception as e:
        print(f"❌ An unexpected error occurred during mask generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)