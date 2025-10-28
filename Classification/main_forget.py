# === START OF main_forget.py (with EWC and KL) ===
import os
import copy
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import arg_parser # REMEMBER TO ADD --ewc_lambda HERE
import utils
from output_utils import *
import datetime
import json
import glob

# --- Function Definitions ---

def test_model_accuracy(model, test_loader, device):
    # (Same as before)
    model.eval(); correct = 0; total = 0
    if test_loader is None: return 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device); output = model(data)
            _, predicted = torch.max(output.data, 1); total += target.size(0); correct += (predicted == target).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0

def apply_mask_with_initial_state(model, mask, initial_state):
    # (Same as before, handles mask=None)
    if mask is None or initial_state is None: return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask and name in initial_state:
                mask_tensor = mask[name].to(param.device); param.data = param.data * mask_tensor + initial_state[name] * (1 - mask_tensor)

# === EWC: Function to Calculate Fisher Diagonal ===
def calculate_fisher_diagonal(model, loader, criterion, device):
    """
    Calculates the empirical Fisher Information Matrix diagonal.
    Uses the average of squared gradients of the CE loss on the retain set.
    """
    print("Calculating Fisher Information Diagonal...")
    fisher_diag = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_diag[name] = torch.zeros_like(param.data)
    
    model.train() # Ensure gradients are computed
    num_samples = 0
    
    # Check if loader is empty
    if len(loader) == 0:
        print("❌ ERROR: Loader is empty for Fisher calculation.")
        return None

    for image, target in loader:
        image, target = image.to(device), target.to(device)
        model.zero_grad()
        
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher_diag:
                # Accumulate squared gradients (element-wise), weighted by batch size
                fisher_diag[name] += (param.grad.data ** 2) * len(target)
        
        num_samples += len(target)
        
    if num_samples == 0:
        print("❌ ERROR: No samples processed during Fisher calculation.")
        return None
        
    # Average the squared gradients
    for name in fisher_diag:
        fisher_diag[name] = fisher_diag[name] / num_samples
        
    print(f"✅ Fisher Diagonal calculation complete (based on {num_samples} samples).")
    return fisher_diag

# === EWC: Function to Calculate EWC Loss ===
def calculate_ewc_loss(current_model, original_state_dict, fisher_diag, device):
    """
    Calculates the EWC loss.
    Loss = sum( fisher_diag[p] * (param[p] - param_original[p])**2 )
    """
    ewc_loss = 0.0
    
    for name, param in current_model.named_parameters():
        if param.requires_grad and name in original_state_dict and name in fisher_diag:
            original_param = original_state_dict[name].to(device)
            fisher_importance = fisher_diag[name].to(device)
            
            if param.shape == original_param.shape and param.shape == fisher_importance.shape:
                # Element-wise multiplication and sum
                ewc_loss += (fisher_importance * (param - original_param) ** 2).sum()
            
    return ewc_loss

# === KL Divergence Function (Fallback) ===
def calculate_kl_divergence(current_model, original_state_dict, device):
    """
    Calculates L2 penalty between parameters (proxy for KL).
    """
    kl_loss = 0.0
    num_params = 0
    for name, param in current_model.named_parameters():
        if param.requires_grad and name in original_state_dict:
            original_param = original_state_dict[name].to(device)
            if param.data.shape == original_param.shape:
                kl_loss += F.mse_loss(param, original_param, reduction='sum')
                num_params += param.numel() 
    return kl_loss if num_params > 0 else torch.tensor(0.0).to(device)


# === Main Unlearning Function (Now supports EWC) ===
def unlearn_with_mask(data_loaders, model, criterion, args, mask, original_state_dict, fisher_diag, experiment_dir, method, percentage):
    """
    Perform unlearning. Includes EWC or KL regularization for FT method when mask=None.
    """
    print(f"\n🎯 Starting unlearning for {method} - {percentage*100:.0f}% forgetting")
    using_mask = mask is not None
    if not using_mask: print("🎭 Running WITHOUT mask.")
    else: print(f"🎭 Running WITH mask (Retention: {calculate_mask_retention(mask):.1f}%).")

    initial_state = {}
    if using_mask:
        print("... Storing initial state for masked parameters ...")
        for name, param in model.named_parameters():
            if name in mask: initial_state[name] = param.data.clone()

    optimizer = torch.optim.SGD(model.parameters(), args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_accuracy = 0; training_log = []

    for epoch in range(args.unlearn_epochs):
        model.train()
        train_loss = 0; total_ce_loss = 0.0; total_reg_loss = 0.0 # Renamed total_kl_loss
        correct = 0; total = 0; processed_batches = 0

        # --- Method-specific training ---
        if method == "GA":
            # ... (GA logic remains unchanged) ...
            data_loader = data_loaders.get("forget")
            if data_loader is None: print("❌ ERROR: No forget loader found for GA method in data_loaders"); break
            for batch_idx, (image, target) in enumerate(data_loader):
                image, target = image.cuda(), target.cuda(); output = model(image)
                loss = -criterion(output, target) * 10.0
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if using_mask:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n in mask and p.grad is not None: p.grad *= mask[n].to(p.device)
                optimizer.step()
                if using_mask: apply_mask_with_initial_state(model, mask, initial_state)
                train_loss += loss.item(); processed_batches += 1
                if processed_batches >= 30: break

        elif method == "FT":
            data_loader = data_loaders.get("retain")
            if data_loader is None: print("❌ ERROR: No retain loader found for FT method"); break

            for batch_idx, (image, target) in enumerate(data_loader):
                image, target = image.cuda(), target.cuda(); output = model(image)
                ce_loss = criterion(output, target)
                
                # --- EWC / KL / Naive Logic ---
                reg_loss = torch.tensor(0.0).to(image.device)
                if not using_mask: # Apply penalty only if no mask
                    if args.ewc_lambda > 0 and fisher_diag is not None:
                        # --- EWC Loss ---
                        reg_loss = calculate_ewc_loss(model, original_state_dict, fisher_diag, image.device)
                        total_loss = ce_loss + args.ewc_lambda * reg_loss
                        total_reg_loss += reg_loss.item()
                    elif args.kl_lambda > 0:
                        # --- KL Loss (Fallback) ---
                        reg_loss = calculate_kl_divergence(model, original_state_dict, image.device)
                        total_loss = ce_loss + args.kl_lambda * reg_loss
                        total_reg_loss += reg_loss.item()
                    else:
                        # --- Naive FT ---
                        total_loss = ce_loss
                else:
                    total_loss = ce_loss # Masked FT (no regularization)
                # --- End Regularization Logic ---

                optimizer.zero_grad(); total_loss.backward()
                if using_mask:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n in mask and p.grad is not None: p.grad *= mask[n].to(p.device)
                optimizer.step()
                if using_mask: apply_mask_with_initial_state(model, mask, initial_state)

                train_loss += total_loss.item(); total_ce_loss += ce_loss.item(); processed_batches += 1
                _, pred = output.max(1); total += target.size(0); correct += pred.eq(target).sum().item()

        elif method == "RL":
            # ... (RL logic remains unchanged) ...
            retain_loader=data_loaders.get("retain"); forget_loader=data_loaders.get("forget")
            if retain_loader is None or forget_loader is None: print("❌ ERROR: Missing retain or forget loader for RL method"); break
            retain_iter=iter(retain_loader); forget_iter=iter(forget_loader); num_batches=min(len(retain_loader), len(forget_loader))
            if num_batches == 0: print("❌ ERROR: Not enough batches for RL method (one loader might be empty)."); break
            total_r_loss, total_f_loss = 0.0, 0.0
            for i in range(num_batches):
                try: # Retain step
                    img_r, tgt_r = next(retain_iter); img_r, tgt_r = img_r.cuda(), tgt_r.cuda()
                    out_r = model(img_r); loss_r = criterion(out_r, tgt_r)
                    optimizer.zero_grad(); loss_r.backward()
                    if using_mask:
                        with torch.no_grad():
                            for n, p in model.named_parameters():
                                if n in mask and p.grad is not None: p.grad *= mask[n].to(p.device)
                    optimizer.step(); total_r_loss += loss_r.item()
                    _, pred = out_r.max(1); total += tgt_r.size(0); correct += pred.eq(tgt_r).sum().item()
                except StopIteration: print("Warning: Retain loader finished early."); break
                try: # Forget step
                    img_f, tgt_f = next(forget_iter); img_f, tgt_f = img_f.cuda(), tgt_f.cuda()
                    out_f = model(img_f); forget_mult = 5.0 
                    loss_f = -criterion(out_f, tgt_f) * forget_mult
                    optimizer.zero_grad(); loss_f.backward()
                    if using_mask:
                        with torch.no_grad():
                            for n, p in model.named_parameters():
                                if n in mask and p.grad is not None: p.grad *= mask[n].to(p.device)
                    optimizer.step(); total_f_loss += loss_f.item()
                except StopIteration: print("Warning: Forget loader finished early."); break
                if using_mask: apply_mask_with_initial_state(model, mask, initial_state)
                processed_batches += 1
            train_loss = (total_r_loss + total_f_loss) / processed_batches if processed_batches > 0 else 0.0

        # --- End Method Logic ---
        avg_loss = train_loss / processed_batches if processed_batches > 0 else 0.0
        avg_ce_loss = total_ce_loss / processed_batches if processed_batches > 0 else 0.0
        avg_reg_loss = total_reg_loss / processed_batches if processed_batches > 0 else 0.0 # Renamed avg_kl_loss

        # --- Evaluation ---
        test_acc = test_model_accuracy(model, data_loaders.get("val"), 'cuda')
        retain_acc = test_model_accuracy(model, data_loaders.get("retain"), 'cuda')
        forget_acc = test_model_accuracy(model, data_loaders.get("forget"), 'cuda')
        if test_acc > best_accuracy: best_accuracy = test_acc

        log_entry = {'epoch': epoch, 'train_loss': avg_loss, 'test_accuracy': test_acc, 'retain_accuracy': retain_acc, 'forget_accuracy': forget_acc}
        if method == "FT" and not using_mask: 
            log_entry['train_ce_loss'] = avg_ce_loss
            log_entry['train_reg_loss'] = avg_reg_loss # Renamed
        training_log.append(log_entry)

        print_str = f'Epoch: {epoch:3d}, Loss: {avg_loss:7.3f}, Test Acc: {test_acc:6.2f}%, Retain Acc: {retain_acc:6.2f}%, Forget Acc: {forget_acc:6.2f}%'
        
        # --- Updated Logging String ---
        if method == "FT" and not using_mask:
            reg_type = "EWC" if args.ewc_lambda > 0 and fisher_diag is not None else "KL" if args.kl_lambda > 0 else "Naive"
            # Use 'EWC' or 'KL' as the label for the regularization loss
            if reg_type != "Naive":
                 print_str += f' (CE: {avg_ce_loss:.3f}, {reg_type}: {avg_reg_loss:.3e})'
            else:
                 print_str += f' (CE: {avg_ce_loss:.3f})' # No reg loss to show for Naive
        # --- End Updated Logging ---
            
        if epoch % 2 == 0 or epoch < 3 or epoch == args.unlearn_epochs - 1: print(print_str)

    # --- Final Evaluation ---
    print("🏁 Unlearning loop finished. Final evaluation:")
    final_test_acc = test_model_accuracy(model, data_loaders.get("val"), 'cuda')
    final_retain_acc = test_model_accuracy(model, data_loaders.get("retain"), 'cuda')
    final_forget_acc = test_model_accuracy(model, data_loaders.get("forget"), 'cuda')
    efficacy_separation = final_retain_acc - final_forget_acc
    print(f"   Final Test Acc: {final_test_acc:.2f}%"); print(f"   Final Retain Acc: {final_retain_acc:.2f}%"); print(f"   Final Forget Acc: {final_forget_acc:.2f}%"); print(f"   Final Efficacy (Ret-For): {efficacy_separation:.2f}%")

    # --- Updated Method Naming ---
    method_name = method # Default
    if method == "FT" and not using_mask:
        if args.ewc_lambda > 0 and fisher_diag is not None:
            method_name = "FT_EWC"
        elif args.kl_lambda > 0:
            method_name = "FT_KL"
        else:
            method_name = "FT_Naive"
    # --- End Method Naming ---

    results = {
        'method': method_name, 'forget_percentage': percentage, 'final_test_accuracy': final_test_acc,
        'final_retain_accuracy': final_retain_acc, 'final_forget_accuracy': final_forget_acc,
        'best_test_accuracy': best_accuracy, 'efficacy_separation': efficacy_separation,
        'training_log': training_log, 'unlearn_epochs': args.unlearn_epochs,
        'mask_retention': calculate_mask_retention(mask),
        'rl_forget_multiplier': None, # Default
        # --- Updated Results Dict ---
        'kl_lambda': args.kl_lambda if method_name == "FT_KL" else None,
        'ewc_lambda': args.ewc_lambda if method_name == "FT_EWC" else None
    }
    
    if method == "RL":
        forget_mult_used = 5.0 # Hardcoded value from RL loop
        results['rl_forget_multiplier'] = forget_mult_used

    return results, model

# --- Other Helper Functions (Unchanged) ---

def calculate_mask_retention(mask):
    if mask is None: return None
    total_p = sum(m.numel() for m in mask.values() if m is not None); retained_p = sum(m.sum().item() for m in mask.values() if m is not None)
    return (retained_p / total_p * 100) if total_p > 0 else 0.0

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
    num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle)

def find_mask_path(method, percent, experiment_dir):
    # (Same as version that includes Fisher name)
    percent_str1 = f"{percent}percent"; percent_str2 = f"{percent}_percent"
    mask_name_orig1 = f"mask_otsu_{method}_{percent}percent_conservative.pt"; mask_name_orig2 = f"mask_otsu_{method}_{percent}percent.pt"; mask_name_fisher = f"mask_fisher_otsu_{method}_{percent}percent.pt"
    possible_paths = [ os.path.join(experiment_dir, 'masks', method, percent_str1, mask_name_fisher), os.path.join(experiment_dir, 'masks', method, percent_str1, mask_name_orig1), os.path.join(experiment_dir, 'masks', method, percent_str2, mask_name_orig1), os.path.join(experiment_dir, 'masks', method, percent_str1, mask_name_orig2), ]
    for mask_path in possible_paths:
        if os.path.exists(mask_path): print(f"Found mask at: {mask_path}"); return mask_path
    print(f"❌ ERROR: No mask file found matching patterns for {method} {percent}%"); return None

def find_existing_experiment_dir(experiment_name, output_base_dir="results"):
    # (Same as previous correct version)
    if not os.path.exists(output_base_dir): return None
    candidates = [os.path.join(output_base_dir, d) for d in os.listdir(output_base_dir) if d.startswith(experiment_name) and os.path.isdir(os.path.join(output_base_dir, d)) and os.path.exists(os.path.join(output_base_dir, d, 'masks'))]
    if not candidates: print(f"❌ No existing experiment directories with masks found starting with '{experiment_name}'"); return None
    candidates.sort(key=os.path.getmtime, reverse=True); selected = candidates[0]
    print(f"🎯 Selected most recent experiment directory: {selected}"); return selected

def save_unlearning_results(results, method, percentage, paths):
    # (Same as previous correct version - uses utils.make_json_serializable)
    try:
        results_dir = os.path.join(paths['unlearning_dir'], method, f"{int(percentage*100)}percent")
        os.makedirs(results_dir, exist_ok=True); results_path = os.path.join(results_dir, 'unlearning_results.json')
        serializable_results = utils.make_json_serializable(results)
        with open(results_path, 'w') as f: json.dump(serializable_results, f, indent=2)
        print(f"💾 Unlearning results saved to: {results_path}")
    except Exception as e: print(f"❌ Error saving unlearning results: {e}")

def get_unlearning_dir(paths, method, percentage):
    # (Same as previous correct version - uses method name for dir)
    unlearning_dir = os.path.join(paths['unlearning_dir'], method, f"{int(percentage*100)}percent")
    os.makedirs(unlearning_dir, exist_ok=True); return unlearning_dir


# === main Function (Loads original model state dict for KL/EWC) ===
def main():
    args = arg_parser.parse_args() # REMEMBER TO ADD --ewc_lambda
    if torch.cuda.is_available(): device = torch.device(f"cuda:{int(args.gpu)}")
    else: device = torch.device("cpu")
    if args.seed: utils.setup_seed(args.seed)
    seed = args.seed

    # --- Directory Logic (Unchanged) ---
    if args.experiment_dir:
        if not os.path.exists(args.experiment_dir): print(f"❌ ERROR: Specified dir not found: {args.experiment_dir}"); return
        print(f"🎯 USING SPECIFIED EXPERIMENT DIRECTORY: {args.experiment_dir}"); experiment_dir = args.experiment_dir
    else:
        print(f"🔍 Looking for existing directory for: {args.experiment_name}"); experiment_dir = find_existing_experiment_dir(args.experiment_name, args.output_base_dir)
        if experiment_dir is None: print("❌ ERROR: Could not find existing dir."); return
    print(f"✅ Using experiment directory: {experiment_dir}")
    paths = {k: os.path.join(experiment_dir, d) for k, d in zip(['experiment_dir', 'masks_dir', 'summary_dir', 'logs_dir', 'unlearning_dir'], ['.', 'masks', 'summary', 'logs', 'unlearning'])}
    paths['experiment_dir'] = experiment_dir
    for k in ['summary_dir', 'logs_dir', 'unlearning_dir']: os.makedirs(paths[k], exist_ok=True)
    # --- End Directory Logic ---

    model, train_loader_full, val_loader, _, _ = utils.setup_model_dataset(args)
    model.cuda()

    # --- Method/Percentage Setup (Unchanged) ---
    if args.unlearn is None: print("❌ ERROR: Please specify --unlearn"); return
    methods_to_run = [args.unlearn]
    if args.forget_percentage is None: print("❌ ERROR: Please specify --forget_percentage"); return
    percentages_to_run = [args.forget_percentage]
    print(f"🔄 Running unlearning for method: {methods_to_run[0]}"); print(f"📊 Testing forgetting percentage: {percentages_to_run[0]*100:.0f}%"); print(f"📁 Using experiment directory: {experiment_dir}")
    # --- End Method/Percentage Setup ---

    # --- Load Original Model AND State Dict (Unchanged) ---
    if not os.path.exists(args.model_path): print(f"❌ ERROR: Model file not found: {args.model_path}"); return
    print(f"📥 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device); state_dict = checkpoint.get("state_dict", checkpoint)
    original_state_dict = {k.replace("module.", ""): v.clone() for k, v in state_dict.items()} # Clone tensors
    new_state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in state_dict.items()])
    model.load_state_dict(new_state_dict, strict=False); print("✅ Model loaded successfully!")
    # --- End Load Model ---

    criterion = nn.CrossEntropyLoss()
    all_results = {}

    # --- Setup Logging (Updated for EWC) ---
    progress_log_path = os.path.join(paths['logs_dir'], 'unlearning_progress.log')
    with open(progress_log_path, 'a') as f:
        f.write(f"\n--- New Unlearning Run: {datetime.datetime.now()} ---\n")
        f.write(f"Method: {methods_to_run[0]}, Percentage: {percentages_to_run[0]*100:.0f}%")
        if args.no_mask:
            if args.unlearn == 'FT':
                if args.ewc_lambda > 0:
                    f.write(f" (--no_mask, EWC lambda={args.ewc_lambda})")
                elif args.kl_lambda > 0:
                    f.write(f" (--no_mask, KL lambda={args.kl_lambda})")
                else:
                    f.write(f" (--no_mask, Naive FT)")
            else:
                 f.write(f" (--no_mask)") # e.g., for RL --no_mask
        f.write("\n")
    # --- End Logging Setup ---

    # --- Main Loop ---
    method = methods_to_run[0]; percentage = percentages_to_run[0]; percent_int = int(percentage * 100)
    print(f"\n{'='*60}\n🔄 RUNNING UNLEARNING FOR {method} at {percent_int}%\n{'='*60}")

    # --- Skip logic (Unchanged) ---
    if (method in ['FT', 'RL'] and percentage == 1.0 and not args.no_mask) or \
       (method == 'GA' and percentage == 0.0):
        print(f"⏭️ Skipping {method} for {percent_int}% - inappropriate data/mask combination.")
        return

    # --- Data Prep (Unchanged) ---
    current_dataset_copy = copy.deepcopy(train_loader_full.dataset)
    marked_dataset = mark_dataset_for_percentage(current_dataset_copy, percentage, seed=seed)
    if marked_dataset is None: print("❌ ERROR: Failed mark dataset."); return
    forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
    forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
    retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
    if (method == 'GA' and forget_loader is None) or \
       (method == 'FT' and retain_loader is None and args.no_mask and args.ewc_lambda == 0) or \
       (method == 'FT' and retain_loader is None and not args.no_mask) or \
       (method == 'RL' and (retain_loader is None or forget_loader is None)):
        print(f"❌ ERROR: Missing required data loader for {method} at {percent_int}%"); return
    unlearn_data_loaders = OrderedDict(retain=retain_loader, forget=forget_loader, val=val_loader)
    # --- End Data Prep ---

    # --- NEW: Calculate Fisher Diagonal (if EWC is selected) ---
    fisher_diag = None
    if method == 'FT' and args.no_mask and args.ewc_lambda > 0:
        if retain_loader is None:
            print("❌ ERROR: Retain loader is required for EWC Fisher calculation but not found.")
            return 
        
        print("...Preparing model for Fisher calculation (using original weights)...")
        # We need a model copy in the *original* state to calculate the Fisher
        original_model_for_fisher = copy.deepcopy(model)
        original_model_for_fisher.load_state_dict(original_state_dict, strict=False)
        original_model_for_fisher.cuda()
        
        fisher_diag = calculate_fisher_diagonal(
            original_model_for_fisher,
            retain_loader, # Calculate Fisher on the retain set
            criterion,
            device
        )
        del original_model_for_fisher # Free memory
        
        if fisher_diag is None:
            print("❌ ERROR: Fisher diagonal calculation failed. Aborting EWC.")
            return 
    # --- End Fisher Calculation ---

    # --- Find Mask (or skip if --no_mask) (Updated) ---
    mask = None
    if not args.no_mask:
        mask_path = find_mask_path(method, percent_int, experiment_dir)
        if mask_path is None: print(f"❌ ERROR: Mask requested but not found."); return
        print(f"📂 Loading mask from: {mask_path}")
        mask = torch.load(mask_path, map_location=device)
        print(f"✅ Mask loaded! Retention: {calculate_mask_retention(mask):.1f}%")
    else:
        print("🎭 Running WITHOUT mask (--no_mask specified).")
        # Updated print logic for EWC/KL/Naive
        if method == 'FT':
            if args.ewc_lambda > 0:
                if fisher_diag is not None:
                    print(f"   Using EWC Regularization with lambda = {args.ewc_lambda}")
                else:
                    print("   WARNING: EWC lambda > 0 but Fisher calculation failed. Running Naive FT.")
            elif args.kl_lambda > 0:
                print(f"   Using KL Regularization with lambda = {args.kl_lambda}")
            else:
                print(f"   Running Naive Finetuning (KL/EWC lambda = 0)")
    # --- End Mask Logic ---

    # --- Perform Unlearning (Updated) ---
    current_model = copy.deepcopy(model); current_model.cuda()
    results, unlearned_model = unlearn_with_mask(
        unlearn_data_loaders, current_model, criterion, args, mask,
        original_state_dict, # Pass original state dict
        fisher_diag,         # Pass Fisher diagonal (or None)
        experiment_dir, method, percentage
    )
    # --- End Unlearning ---

    # --- Save Results & Model (Updated) ---
    if results:
        current_method_name = results['method'] # Get updated name (FT_EWC, FT_KL, etc.)
        save_unlearning_results(results, current_method_name, percentage, paths)

        unlearning_dir = get_unlearning_dir(paths, current_method_name, percentage)
        model_save_path = os.path.join(unlearning_dir, 'model_unlearned.pth.tar')
        torch.save({'state_dict': unlearned_model.state_dict(), 'accuracy': results['final_test_accuracy'], 'epoch': args.unlearn_epochs, 'method': current_method_name, 'percentage': percentage}, model_save_path)
        print(f"💾 Unlearned model saved to: {model_save_path}")

        if current_method_name not in all_results: all_results[current_method_name] = {}
        all_results[current_method_name][str(percentage)] = results

        with open(progress_log_path, 'a') as f:
            f.write(f"COMPLETED {current_method_name}_{percent_int}%: Test={results['final_test_accuracy']:.2f}%, Retain={results['final_retain_accuracy']:.2f}%, Forget={results['final_forget_accuracy']:.2f}%, Efficacy={results['efficacy_separation']:.2f}%\n")
    else:
        # Updated failed log message
        log_method_name = method
        if method == "FT" and not args.no_mask:
            if args.ewc_lambda > 0:
                log_method_name = "FT_EWC"
            elif args.kl_lambda > 0:
                log_method_name = "FT_KL"
            else:
                log_method_name = "FT_Naive"
        print(f"❌ Unlearning failed for {log_method_name} at {percent_int}%.")
        with open(progress_log_path, 'a') as f: f.write(f"FAILED {log_method_name}_{percent_int}%\n")
    # --- End Save Results ---

    # --- Update Comprehensive Summary (Unchanged) ---
    summary_path = os.path.join(paths['summary_dir'], 'unlearning_comprehensive_summary.json')
    existing_summary = {};
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f: existing_summary = json.load(f)
        except json.JSONDecodeError: print(f"⚠️ Overwriting corrupt summary: {summary_path}")
    for m, p_results in all_results.items():
        if m not in existing_summary: existing_summary[m] = {}
        existing_summary[m].update(p_results)
    try:
        serializable_summary = utils.make_json_serializable(existing_summary)
        with open(summary_path, 'w') as f: json.dump(serializable_summary, f, indent=2)
        print(f"📊 Comprehensive summary updated: {summary_path}")
    except Exception as e: print(f"❌ Error saving comprehensive summary: {e}")
    # --- End Summary Update ---

    # --- Print Final Summary Table (Unchanged) ---
    print(f"\n{'='*60}\n📈 UNLEARNING RUN COMPLETE\n{'='*60}"); print(f"📁 Results saved in: {experiment_dir}")
    print("\n📋 FINAL RESULTS (from comprehensive summary):")
    print("Method         | Percentage | Test Acc | Retain Acc | Forget Acc | Efficacy (Ret-For)")
    print("-" * 85)
    sorted_methods = sorted(existing_summary.keys())
    for m in sorted_methods:
        sorted_percentages = sorted(existing_summary[m].keys(), key=float)
        for p_str in sorted_percentages:
            try:
                res = existing_summary[m][p_str]; p_float = float(p_str); p_int = int(p_float * 100)
                print(f"{m:13} | {p_int:>4}%       | {res.get('final_test_accuracy', -1):>8.2f}% | "
                      f"{res.get('final_retain_accuracy', -1):>10.2f}% | {res.get('final_forget_accuracy', -1):>9.2f}% | "
                      f"{res.get('efficacy_separation', -1):>18.2f}%")
            except Exception as e: print(f"Error displaying result for {m} {p_str}: {e}")
    # --- End Final Print ---

if __name__ == "__main__":
    # --- Fallback (Unchanged) ---
    if not hasattr(utils, 'make_json_serializable'):
         print("Warning: utils.make_json_serializable not found. Saving summary might fail.")
         def basic_serializable(obj):
             import numpy as np; import torch
             if isinstance(obj, (np.ndarray, torch.Tensor)): return obj.tolist()
             if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
             if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
             if isinstance(obj, (np.bool_)): return bool(obj)
             if isinstance(obj, dict): return {k: basic_serializable(v) for k, v in obj.items()}
             if isinstance(obj, (list, tuple)): return [basic_serializable(i) for i in obj]
             return obj
         utils.make_json_serializable = basic_serializable
    main()
# === END OF main_forget.py ===