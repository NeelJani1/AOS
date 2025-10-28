# === START OF REVISED generate_masks.py (Fisher-Weighted Saliency) ===
import os
import copy
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import arg_parser # Your custom argument parser file
import utils
from otsu_utils import *
from output_utils import *
import datetime
import json

# --- Fisher Diagonal Computation ---
def compute_fisher_diagonal(model, loader, criterion, device, epsilon=1e-8):
    """
    Computes the empirical diagonal Fisher Information Matrix.
    Uses the squared gradients averaged over the provided data loader.
    """
    print(f"🔧 Computing Empirical Fisher Diagonal using {len(loader.dataset)} samples...")
    fisher_diag = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diag[name] = torch.zeros_like(param, device=device)

    model.eval() # Ensure model is in eval mode (no dropout, etc.)
    num_samples = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        num_samples += batch_size

        output = model(data)
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared gradients, scaled by batch size if averaging later
                    fisher_diag[name] += param.grad.data.pow(2) * batch_size # Weighted sum

        if batch_idx >= 10: # Limit batches for faster estimation if needed
             print(f"   (Stopping Fisher estimation early after {batch_idx+1} batches for speed)")
             break


    # Average the squared gradients over the total number of samples processed
    if num_samples > 0:
        with torch.no_grad():
            for name in fisher_diag:
                fisher_diag[name] /= num_samples
                # Apply damping for numerical stability before sqrt/inverse
                fisher_diag[name] += epsilon
    else:
        print("❌ ERROR: No samples processed during Fisher computation.")
        return None # Indicate failure

    print(f"✅ Fisher Diagonal computed over {num_samples} samples.")
    return fisher_diag


# --- Other Helper Functions (scale_ga, debug_stats, get_otsu, etc.) ---
# (Keep these exactly as they were in the previous version)
def scale_ga_gradients(gradients, scale_factor=0.5):
    # (Same as before)
    print(f"🔧 Scaling GA gradients by {scale_factor}x")
    with torch.no_grad():
        for name in gradients:
            if gradients[name] is not None and not isinstance(gradients[name], float):
                gradients[name] = gradients[name] * scale_factor
    return gradients

def debug_gradient_stats(gradients, method):
    # (Same as before)
    if not gradients: return None
    all_grads_list = [g.flatten().cpu().numpy() for g in gradients.values() if g is not None and not isinstance(g, float)]
    if not all_grads_list: return None
    all_grads = np.concatenate(all_grads_list)
    print(f"🔍 DEBUG: {method} Gradient Stats (Count: {len(all_grads):,})")
    print(f"   Min: {all_grads.min():.6e}, Max: {all_grads.max():.6e}, Mean: {all_grads.mean():.6e}, Median: {np.median(all_grads):.6e}")
    return all_grads

def get_otsu_method(method, **kwargs):
    # (Same as before - uses bounded/fixed logic)
    if method == 'bounded':
        min_r, max_r = kwargs.get('min_retention', 0.1), kwargs.get('max_retention', 0.3)
        unlearn_m = kwargs.get('unlearn_method', 'RL')
        def bounded_otsu(grads):
            # Pass only valid tensors to Otsu
            valid_grads = {k:v for k,v in grads.items() if v is not None and not isinstance(v, float)}
            if not valid_grads: return {}
            thresh = enhanced_otsu_threshold(valid_grads, method=unlearn_m, target_retention_range=(min_r, max_r))
            mask = {n: (g.abs() >= thresh).float() for n, g in valid_grads.items()}
            return mask
        return bounded_otsu
    elif method == 'fixed':
        ret_rate = kwargs.get('retention_rate', 0.2)
        def fixed_otsu(grads):
            mask = {}
            for n, g in grads.items():
                if g is None or isinstance(g, float): continue
                flat_g = g.abs().flatten()
                n_el = len(flat_g)
                if n_el == 0: mask[n] = torch.zeros_like(g); continue
                k = int(n_el * ret_rate)
                if k > 0:
                    idx = max(1, n_el - k)
                    thresh = torch.kthvalue(flat_g, idx)[0]
                    mask[n] = (g.abs() >= thresh).float()
                else: mask[n] = torch.zeros_like(g)
            return mask
        return fixed_otsu
    else: # Default (should not be hit if using bounded)
        def default_otsu(grads):
            valid_grads = {k:v for k,v in grads.items() if v is not None and not isinstance(v, float)}
            if not valid_grads: return {}
            thresh = enhanced_otsu_threshold(valid_grads)
            mask = {n: (g.abs() >= thresh).float() for n, g in valid_grads.items()}
            return mask
        return default_otsu


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REVISED MASK GENERATION FUNCTION (with Fisher Weighting) <<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_masks_for_percentage(data_loaders, model, criterion, args, forget_percentage, paths):
    """
    Generate Fisher-Weighted Otsu mask.
    Uses abs(Fisher_diag^(-1/2) * forget_grad) as saliency signal.
    """
    optimizer = torch.optim.SGD(model.parameters(), args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    forget_loader = data_loaders.get("forget")
    retain_loader = data_loaders.get("retain") # Needed only for Fisher calculation if desired
    model.eval()

    print(f"\n🎯 Generating Fisher-Weighted mask for {forget_percentage*100:.0f}% forgetting")
    print(f"🔧 METHOD (for logging): {args.unlearn}") # Method name used for file paths etc.

    # --- Step 1: Compute Fisher Diagonal ---
    # Use forget_loader to estimate Fisher relevant to the data being forgotten
    if forget_loader is None:
         print("❌ ERROR: Forget loader is required to compute Fisher diagonal.")
         return None, 0
    fisher_diag = compute_fisher_diagonal(model, forget_loader, criterion, 'cuda')
    if fisher_diag is None:
        print("❌ ERROR: Fisher diagonal computation failed.")
        return None, 0

    # --- Step 2: Compute Raw Forget Gradient ---
    # We need the raw gradient vector g = grad(loss_forget)
    print("🔧 Computing raw forget gradients (using positive loss)...")
    raw_forget_gradients = {name: None for name, p in model.named_parameters() if p.requires_grad}
    batch_count_f = 0
    for i, (image, target) in enumerate(forget_loader):
        image, target = image.cuda(), target.cuda()
        output = model(image)
        loss = criterion(output, target) # Use POSITIVE loss for raw gradient
        optimizer.zero_grad(); loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if raw_forget_gradients.get(name) is None: raw_forget_gradients[name] = param.grad.data.clone()
                    else: raw_forget_gradients[name] += param.grad.data
        batch_count_f += 1
        if batch_count_f >= 5: break # Limit batches

    # Average the raw forget gradients
    if batch_count_f > 0:
        with torch.no_grad():
            for name in raw_forget_gradients:
                if raw_forget_gradients[name] is not None:
                    raw_forget_gradients[name] /= batch_count_f
    else:
        print("❌ ERROR: No forget batches processed for gradient computation.")
        return None, 0

    # --- Step 3: Compute Fisher-Scaled Gradient Magnitude ---
    print("🔧 Computing Fisher-scaled gradient magnitudes...")
    scaled_gradient_magnitudes = {}
    with torch.no_grad():
        for name in raw_forget_gradients:
            grad = raw_forget_gradients.get(name)
            fisher = fisher_diag.get(name)
            if grad is not None and fisher is not None:
                # s = grad / sqrt(fisher + epsilon) - Epsilon already added in compute_fisher
                scaled_grad = grad / fisher.sqrt()
                scaled_gradient_magnitudes[name] = torch.abs_(scaled_grad)
            else:
                # Keep None if either grad or fisher was missing
                 scaled_gradient_magnitudes[name] = None

    # --- Step 4: Clipping (Optional but recommended) ---
    print(f"🔧 Pre-processing Fisher-scaled grads (clipping)...")
    # Use the scaled magnitudes for clipping calculation
    valid_scaled_grads = [g.detach().cpu().float().flatten() for g in scaled_gradient_magnitudes.values() if g is not None]
    if valid_scaled_grads:
        all_scaled_grads_flat = torch.cat(valid_scaled_grads)
        if all_scaled_grads_flat.numel() > 0:
            clip_value = torch.quantile(all_scaled_grads_flat, 0.999) # Clip high magnitudes
            print(f"     Clipping scaled MAGNITUDES at 99.9th percentile: {clip_value.item():.4e}")
            with torch.no_grad():
                for name in scaled_gradient_magnitudes:
                    if scaled_gradient_magnitudes[name] is not None:
                        p99_9 = clip_value.to(scaled_gradient_magnitudes[name].device)
                        # Clamp the magnitude (already positive)
                        torch.clamp_(scaled_gradient_magnitudes[name], max=p99_9)
        else: print("     No scaled gradient values found for clipping.")
    else: print("   No valid scaled gradients found to clip.")
    # --- End Clipping ---


    # --- Step 5: Otsu Thresholding on Scaled Magnitudes ---
    # Debug stats on the values Otsu will see
    _ = debug_gradient_stats(scaled_gradient_magnitudes, f"{args.unlearn}_FisherScaled")

    otsu_method = 'bounded'
    # Use STRICT bounds regardless of method name passed in args (as per our findings)
    print(f"🔧 Applying STRICT Fisher-Weighted Mask Strategy (10-30% bounds)")
    otsu_kwargs = {'min_retention': 0.1, 'max_retention': 0.3, 'unlearn_method': args.unlearn} # Pass unlearn_method for context if needed by Otsu func

    print(f"🔧 Using {otsu_method} Otsu on Fisher-scaled magnitudes")
    otsu_function = get_otsu_method(otsu_method, **otsu_kwargs)

    # Pass the scaled magnitudes to Otsu
    valid_scaled_magnitudes = {k: v for k, v in scaled_gradient_magnitudes.items() if v is not None}
    if not valid_scaled_magnitudes:
        print("❌ ERROR: No valid Fisher-scaled magnitudes to generate mask.")
        return None, 0

    hard_dict = otsu_function(valid_scaled_magnitudes)

    # --- Fallback Logic ---
    if hard_dict is None or not hard_dict:
        print("❌ ERROR: Otsu returned None/empty mask, using fallback (20% fixed)")
        otsu_function = get_otsu_method('fixed', retention_rate=0.2)
        hard_dict = otsu_function(valid_scaled_magnitudes)

    # --- Calculate Retention ---
    total_params, retained_params = 0, 0
    for name, param in model.named_parameters():
         if param.requires_grad:
             total_params += param.numel()
             if name in hard_dict and hard_dict[name] is not None:
                  retained_params += hard_dict[name].sum().item()
    if total_params == 0: return None, 0
    retention_rate = retained_params / total_params * 100

    # --- Low Retention Fallback ---
    if retention_rate < 1.0:
        print(f"⚠️ Low Retention ({retention_rate:.1f}%) - using fallback (20% fixed)")
        fallback_otsu_func = get_otsu_method('fixed', retention_rate=0.2)
        hard_dict = fallback_otsu_func(valid_scaled_magnitudes)
        retained_params = 0 # Recalculate
        for name, param in model.named_parameters():
            if param.requires_grad and name in hard_dict and hard_dict[name] is not None:
                 retained_params += hard_dict[name].sum().item()
        retention_rate = retained_params / total_params * 100
        print(f"🔄 Applied fallback: {retention_rate:.1f}% retention")

    print(f"📊 Fisher Mask Stats: Retention={retention_rate:.1f}%, Method={args.unlearn}")

    # --- Final Mask Assembly ---
    final_mask = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask_tensor = hard_dict.get(name)
            if mask_tensor is not None:
                final_mask[name] = mask_tensor.to(param.device)
            else:
                print(f"⚠️ No mask generated for {name}, creating zero mask (freeze).")
                final_mask[name] = torch.zeros_like(param, device=param.device)

    return final_mask, retention_rate
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> END OF REVISED MASK GENERATION FUNCTION <<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# --- Other Helper Functions (mark_dataset, split_marked, replace_loader) ---
# (Keep these exactly as they were in the previous correct version)
def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
    # (Same as previous correct version)
    np.random.seed(seed)
    try: targets_np = np.array(dataset.targets)
    except AttributeError:
        try: targets_np = np.array([s[1] for s in dataset.samples])
        except AttributeError: print("❌ ERROR: Dataset missing 'targets'/'samples'."); return None
    total_samples = len(targets_np)
    num_to_forget = int(forget_percentage * total_samples)
    if num_to_forget > total_samples: num_to_forget = total_samples
    if num_to_forget < 0: num_to_forget = 0
    forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False) if num_to_forget > 0 else np.array([], dtype=int)
    print(f"🎯 Marking {num_to_forget} samples ({forget_percentage*100:.0f}%) for forgetting")
    if hasattr(dataset, 'targets'):
        dataset_targets = list(dataset.targets)
        for idx in forget_indices:
            if 0 <= idx < len(dataset_targets):
                current_target = dataset_targets[idx]
                if isinstance(current_target, (int, np.integer)) and current_target >= 0: dataset_targets[idx] = -current_target - 1
        dataset.targets = dataset_targets
    elif hasattr(dataset, 'samples'):
        dataset_samples = list(dataset.samples)
        for idx in forget_indices:
             if 0 <= idx < len(dataset_samples):
                path, current_target = dataset_samples[idx]
                if isinstance(current_target, (int, np.integer)) and current_target >= 0: dataset_samples[idx] = (path, -current_target - 1)
        dataset.samples = dataset_samples
        if hasattr(dataset, 'imgs'): dataset.imgs = dataset.samples
        if hasattr(dataset, 'targets'): dataset.targets = [s[1] for s in dataset.samples]
    else: print("❌ ERROR: Unknown dataset structure."); return None
    if hasattr(dataset, 'targets'): final_targets = np.array(dataset.targets); forget_count=np.sum(final_targets < 0); retain_count=np.sum(final_targets >= 0)
    elif hasattr(dataset, 'samples'): final_targets = np.array([s[1] for s in dataset.samples]); forget_count=np.sum(final_targets < 0); retain_count=np.sum(final_targets >= 0)
    else: forget_count=0; retain_count=total_samples
    print(f"✅ Marking complete: {forget_count} forget samples, {retain_count} retain samples")
    return dataset

def split_marked_dataset(marked_dataset):
    # (Same as previous correct version)
    forget_dataset = copy.copy(marked_dataset); retain_dataset = copy.copy(marked_dataset)
    if hasattr(marked_dataset, 'data') and hasattr(marked_dataset, 'targets'):
        targets = np.array(marked_dataset.targets); forget_idx = np.where(targets < 0)[0]; retain_idx = np.where(targets >= 0)[0]
        forget_dataset.data = marked_dataset.data[forget_idx, ...]; forget_dataset.targets = (-targets[forget_idx] - 1).tolist()
        retain_dataset.data = marked_dataset.data[retain_idx, ...]; retain_dataset.targets = targets[retain_idx].tolist()
    elif hasattr(marked_dataset, 'samples'):
        samples = marked_dataset.samples; forget_samples = [(p, -t - 1) for p, t in samples if t < 0]; retain_samples = [(p, t) for p, t in samples if t >= 0]
        forget_dataset.samples, forget_dataset.imgs = forget_samples, forget_samples; retain_dataset.samples, retain_dataset.imgs = retain_samples, retain_samples
        if hasattr(forget_dataset, 'targets'): forget_dataset.targets = [s[1] for s in forget_samples]
        if hasattr(retain_dataset, 'targets'): retain_dataset.targets = [s[1] for s in retain_samples]
    else: print("❌ ERROR: Cannot split dataset."); return None, None
    return forget_dataset, retain_dataset

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    # (Same as previous correct version)
    if dataset is None or len(dataset) == 0: return None
    utils.setup_seed(seed)
    # Adjust num_workers based on system capabilities if needed
    num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle)


# --- Main Function ---
def main():
    # (Keep the main function exactly as it was in the previous correct version)
    # It correctly creates a new directory and loops through methods/percentages
    # calling the REVISED generate_masks_for_percentage function.
    args = arg_parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: CUDA not available, running on CPU. This will be very slow.")
    if args.seed: utils.setup_seed(args.seed)
    seed = args.seed

    print(f"🚀 Creating NEW experiment directory for mask generation: {args.experiment_name}")
    paths = setup_experiment_paths(args)
    experiment_dir = paths['experiment_dir']

    save_experiment_config(args, experiment_dir)
    print(f"✅ Experiment directory created: {experiment_dir}")
    print(f"📄 Config saved in: {experiment_dir}")

    # Use a dummy criterion if only needed for Fisher, avoids issues if model output complex
    criterion_fisher = nn.CrossEntropyLoss() # Standard CE for Fisher
    criterion_mask_grad = nn.CrossEntropyLoss() # Standard CE for FT/RL mask grads

    model, train_loader_full, val_loader, _, _ = utils.setup_model_dataset(args)
    model.to(device) # Move model to device

    methods = ['GA', 'FT', 'RL']
    # forget_percentages = [0.1, 0.5, 0.9] # Test a smaller range first?
    forget_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    if not os.path.exists(args.model_path): print(f"❌ ERROR: Model file not found: {args.model_path}"); return
    print(f"📥 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in state_dict.items()])
    # Load state dict after moving model to device
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Model loaded successfully!")


    results = {}
    progress_log_path = os.path.join(paths['logs_dir'], 'mask_generation_progress.log')
    with open(progress_log_path, 'w') as f: f.write(f"Mask Generation Log (Fisher-Weighted)\n{'='*50}\nStarted: {datetime.datetime.now()}\n\n")

    for method in methods:
        print(f"\n{'='*60}\n🔄 GENERATING MASKS FOR METHOD: {method}\n{'='*60}")
        args.unlearn = method # Critical: Set args.unlearn for generate_masks function
        method_results = {}

        for percentage in forget_percentages:
            print(f"\n📊 Processing {percentage*100:.0f}% forgetting...")

            current_dataset_copy = copy.deepcopy(train_loader_full.dataset)
            marked_dataset = mark_dataset_for_percentage(current_dataset_copy, percentage, seed=seed)
            if marked_dataset is None: continue

            forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
            # Use smaller batch size for Fisher/gradient calculation if memory is tight
            grad_batch_size = min(args.batch_size, 128) # Example adjustment
            forget_loader = replace_loader_dataset(forget_dataset, grad_batch_size, seed=seed, shuffle=False) # No shuffle needed for grads
            retain_loader = replace_loader_dataset(retain_dataset, grad_batch_size, seed=seed, shuffle=False) # No shuffle needed

            skip = False
            # Check for loader existence *before* calling generate_masks
            if method == 'GA' and forget_loader is None:
                 if percentage == 0.0: print(f"⏭️ Skipping GA for 0%"); skip = True
                 else: print(f"❌ ERROR: GA needs forget data"); skip = True
            # Fisher calculation needs forget_loader, so skip if None for all methods if using Fisher
            if forget_loader is None and percentage != 0.0:
                 print(f"❌ ERROR: Forget loader required for Fisher calc, skipping {method} at {percentage*100:.0f}%"); skip = True

            # FT/RL checks (retain loader needed for FT grads if not using Fisher method)
            # If using Fisher method for all, only forget_loader matters for mask gen
            if method in ['FT'] and retain_loader is None: # Keep check for FT if using retain grads
                 if percentage == 1.0: print(f"⏭️ Skipping {method} for 100%"); skip = True
                 else: print(f"❌ ERROR: {method} needs retain data"); skip = True
            # RL needs both if using diff method, only forget if using forget_grad method
            # If using Fisher based on forget_loader, RL doesn't need retain_loader for mask gen

            if skip: continue

            unlearn_data_loaders = OrderedDict(retain=retain_loader, forget=forget_loader, val=val_loader)

            # Use appropriate criterion for mask generation steps
            current_criterion = criterion_mask_grad # Use standard CE for raw grads

            mask, retention_rate = generate_masks_for_percentage(
                unlearn_data_loaders, model, current_criterion, args, percentage, paths
            )

            if mask is None: print(f"❌ Failed mask gen for {method} at {percentage*100:.0f}%"); continue

            mask_dir = os.path.join(paths['masks_dir'], method, f"{int(percentage*100)}percent")
            os.makedirs(mask_dir, exist_ok=True)
            # Use a new name indicating Fisher weighting?
            mask_filename = f"mask_fisher_otsu_{method}_{int(percentage*100)}percent.pt"
            mask_path = os.path.join(mask_dir, mask_filename)
            torch.save(mask, mask_path)

            method_results[percentage] = {
                'retention_rate': retention_rate, 'mask_path': mask_path,
                'forget_samples': len(forget_dataset) if forget_dataset else 0,
                'retain_samples': len(retain_dataset) if retain_dataset else 0
            }
            print(f"💾 Fisher-Weighted Mask saved to: {mask_path}")
            with open(progress_log_path, 'a') as f: f.write(f"{method}_{int(percentage*100)}% (Fisher): {retention_rate:.1f}% retention\n")

        results[method] = method_results

    # --- Save & Print Summary ---
    summary_path = os.path.join(paths['summary_dir'], 'mask_generation_summary_fisher.json') # New summary name
    # Ensure serializable before saving
    serializable_results = utils.make_json_serializable(results)
    with open(summary_path, 'w') as f: json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}\n📈 FISHER MASK GENERATION SUMMARY\n{'='*60}")
    print(f"📁 Results saved in: {experiment_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"📝 Log: {progress_log_path}")

    for method in methods: # Print summary table
        print(f"\n{method} Method (Fisher-Weighted):")
        print("Percentage | Retention | Forget Samples | Retain Samples")
        print("-" * 55)
        # Use sorted keys and handle potential skips/fails
        all_percentages_in_loop = forget_percentages
        method_data = results.get(method, {})
        for percentage in all_percentages_in_loop:
             if percentage in method_data:
                 data = method_data[percentage]
                 print(f"{percentage*100:>4.0f}%      | {data['retention_rate']:>8.1f}% | {data['forget_samples']:>13} | {data['retain_samples']:>12}")
             else:
                 total_samples = len(train_loader_full.dataset)
                 num_forget = int(percentage * total_samples); num_retain = total_samples - num_forget
                 status = 'SKIPPED' if (percentage == 0.0 and method == 'GA') or \
                                      (percentage == 1.0 and method in ['FT', 'RL']) else 'FAILED'
                 print(f"{percentage*100:>4.0f}%      | {status:>8} | {num_forget:>13} | {num_retain:>12}")


if __name__ == "__main__":
    # Add the fallback for make_json_serializable if needed
    if not hasattr(utils, 'make_json_serializable'):
         print("Warning: utils.make_json_serializable not found. Saving summary might fail.")
         def basic_serializable(obj):
              import numpy as np; import torch # Keep imports local
              if isinstance(obj, (np.ndarray, torch.Tensor)): return obj.tolist()
              if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
              if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
              if isinstance(obj, (np.bool_)): return bool(obj)
              if isinstance(obj, dict): return {k: basic_serializable(v) for k, v in obj.items()}
              if isinstance(obj, (list, tuple)): return [basic_serializable(i) for i in obj]
              return obj
         utils.make_json_serializable = basic_serializable
    main()
# === END OF REVISED generate_masks.py ===