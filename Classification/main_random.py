import os
import copy
import numpy as np
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import utils

# Import Otsu utilities
from otsu_utils import *

def scale_ga_gradients(gradients, scale_factor=1000):
    """
    Scale GA gradients to make them comparable to FT/RL
    """
    print(f"🔧 Scaling GA gradients by {scale_factor}x")
    with torch.no_grad():
        for name in gradients:
            gradients[name] = gradients[name] * scale_factor
    return gradients

def save_gradient_ratio(data_loaders, model, criterion, args):
    """
    Generate Otsu mask with method-specific gradient computation
    Supports: GA (Gradient Ascent), FT (Fine-Tuning), RL (Retain Learning)
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    
    model.eval()

    # Initialize gradients dictionary
    for name, param in model.named_parameters():
        gradients[name] = 0

    print(f"\n🔧 METHOD DEBUG: Using unlearn method: {args.unlearn}")

    # METHOD-SPECIFIC GRADIENT COMPUTATION
    if args.unlearn == "GA":
        print("🔧 METHOD: GA (Gradient Ascent) - Maximizing loss on forget set")
        # Gradient Ascent: maximize loss on forget set
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = -criterion(output_clean, target)  # Negative for ascent

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

        # 🆕 SCALE GA GRADIENTS
        gradients = scale_ga_gradients(gradients, scale_factor=1000)

    elif args.unlearn == "FT":
        print("🔧 METHOD: FT (Fine-Tuning) - Standard training on retain set")
        # Fine-Tuning: standard training on retain set
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = criterion(output_clean, target)  # Positive for descent

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

    elif args.unlearn == "RL":
        print("🔧 METHOD: RL (Retain Learning) - SalUn approach")
        # Retain Learning: SalUn approach - balance forgetting and retaining
        # This computes |g_forget| - |g_retain| as in original SalUn paper
        
        # First, compute forget gradients (negative for ascent)
        forget_gradients = {}
        for name, param in model.named_parameters():
            forget_gradients[name] = 0
            
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = -criterion(output_clean, target)  # Negative for ascent

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        forget_gradients[name] += param.grad.data
        
        # Second, compute retain gradients (positive for descent)
        retain_gradients = {}
        for name, param in model.named_parameters():
            retain_gradients[name] = 0
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = criterion(output_clean, target)  # Positive for descent

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        retain_gradients[name] += param.grad.data
        
        # Combine: |g_forget| - |g_retain| as per SalUn paper
        with torch.no_grad():
            for name in gradients:
                forget_abs = torch.abs_(forget_gradients[name])
                retain_abs = torch.abs_(retain_gradients[name])
                gradients[name] = forget_abs - retain_abs

    else:
        print(f"❌ ERROR: Unknown unlearn method: {args.unlearn}")
        print("Available methods: GA, FT, RL")
        return

    # Convert to absolute values for saliency (except RL which already has signed differences)
    with torch.no_grad():
        if args.unlearn != "RL":  # RL already has the SalUn difference
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

    # 🆕 DEBUG: Print gradient statistics
    all_grads = debug_gradient_stats(gradients, args.unlearn)

    # --- ADVANCED OTSU IMPLEMENTATION ---
    print("\n" + "="*60)
    print(f"Advanced Otsu Mask Generation - {args.unlearn} Method")
    print("="*60)
    
    # 🆕 METHOD-SPECIFIC OTSU PARAMETERS
    otsu_method = getattr(args, 'otsu_method', 'conservative')
    
    # Choose Otsu method and parameters based on the unlearning method
    if args.unlearn == "GA":
        # GA needs special handling due to small gradients
        if all_grads is not None and np.percentile(all_grads, 99) < 1e-6:
            print("🔧 GA gradients too small, using fixed retention")
            otsu_method = 'fixed'
            otsu_kwargs = {'retention_rate': 0.5}
        else:
            otsu_method = 'bounded'
            otsu_kwargs = {
                'min_retention': 0.3,
                'max_retention': 0.7
            }
    elif args.unlearn == "RL":
        # RL works well with conservative Otsu
        otsu_method = 'conservative'
        otsu_kwargs = {'conservatism': 0.3}
    else:  # FT
        # FT uses standard parameters
        if otsu_method == 'conservative' or otsu_method == 'layer_aware':
            otsu_kwargs = {'conservatism': getattr(args, 'otsu_conservatism', 0.3)}
        elif otsu_method == 'bounded':
            otsu_kwargs = {
                'min_retention': getattr(args, 'otsu_min_retention', 0.3),
                'max_retention': getattr(args, 'otsu_max_retention', 0.6)
            }
        else:
            otsu_kwargs = {}

    print(f"Using Otsu method: {otsu_method} with parameters: {otsu_kwargs}")
    
    # Generate the mask using selected method
    otsu_function = get_otsu_method(otsu_method, **otsu_kwargs)
    hard_dict = otsu_function(gradients)
    
    # Calculate and print final statistics
    total_params = sum(mask.numel() for mask in hard_dict.values())
    retained_params = sum(mask.sum().item() for mask in hard_dict.values())
    retention_rate = retained_params / total_params * 100
    
    # 🆕 RETENTION RATE FALLBACK
    if retention_rate < 1.0:  # If less than 1% retention
        print(f"⚠️  Warning: {retention_rate:.1f}% retention - using fallback (50%)")
        # Use fixed retention fallback
        fallback_retention = 0.5  # 50% retention
        for name in hard_dict:
            hard_dict[name] = (torch.rand_like(hard_dict[name]) < fallback_retention).float()
        
        # Recalculate statistics
        retained_params = sum(mask.sum().item() for mask in hard_dict.values())
        retention_rate = retained_params / total_params * 100
        print(f"🔄 Applied fallback: {retention_rate:.1f}% retention")
    
    print(f"\n📊 Final Mask Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Retained parameters: {retained_params:,}")
    print(f"   Retention rate: {retention_rate:.1f}%")
    print(f"   Method: {args.unlearn}")
    print(f"   Otsu variant: {otsu_method}")
    
    # Save the adaptive mask
    mask_filename = f"mask_otsu_{args.unlearn}_{otsu_method}.pt"
    save_path = os.path.join(args.save_dir, mask_filename)
    torch.save(hard_dict, save_path)
    print(f"\n💾 Otsu mask saved to: {save_path}")
    print("="*60)


def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
    """
    Mark a portion of the dataset for forgetting by setting their targets to negative
    """
    np.random.seed(seed)
    
    # Get current targets
    targets = np.array(dataset.targets)
    
    # Calculate number of samples to forget based on percentage
    total_samples = len(dataset)
    num_to_forget = int(forget_percentage * total_samples)
    
    # Randomly select samples to forget
    forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False)
    
    print(f"🎯 Marking {num_to_forget} samples ({forget_percentage*100:.0f}%) for forgetting (seed: {seed})")
    
    # Mark forget samples by making targets negative
    for idx in forget_indices:
        dataset.targets[idx] = -dataset.targets[idx] - 1
    
    # Verify marking
    targets_after = np.array(dataset.targets)
    forget_count = np.sum(targets_after < 0)
    retain_count = np.sum(targets_after >= 0)
    
    print(f"✅ Marking complete: {forget_count} forget samples, {retain_count} retain samples")
    
    return dataset


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    # Get forget percentage from args (default to 0.1 if not specified)
    forget_percentage = getattr(args, 'forget_percentage', 0.1)
    
    # Create save directory with percentage info
    save_dir = f"{args.save_dir}_{args.unlearn}_{int(forget_percentage*100)}percent"
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    
    # Prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        if len(dataset) == 0:
            return None
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    # FIXED DATASET SPLITTING WITH PROPER MARKING FOR SPECIFIC PERCENTAGE
    print("🔧 Setting up dataset for unlearning...")
    
    # Create a copy of the training dataset
    original_dataset = copy.deepcopy(train_loader_full.dataset)
    
    # Mark the dataset for unlearning with specific percentage
    marked_dataset = mark_dataset_for_percentage(original_dataset, forget_percentage, seed=seed)
    
    # Now split into forget and retain datasets
    forget_dataset = copy.deepcopy(marked_dataset)
    retain_dataset = copy.deepcopy(marked_dataset)
    
    # Get targets as numpy array
    targets = np.array(forget_dataset.targets)
    
    # Create masks for forget and retain sets
    forget_mask = targets < 0
    retain_mask = targets >= 0
    
    print(f"📊 Dataset split: {forget_mask.sum()} forget samples, {retain_mask.sum()} retain samples")
    
    # Check if we have valid splits
    if forget_mask.sum() == 0 and args.unlearn == "GA":
        print("❌ ERROR: No forget samples found after marking for GA method")
        return
    if retain_mask.sum() == 0 and args.unlearn in ["FT", "RL"]:
        print("❌ ERROR: No retain samples found after marking for FT/RL method")
        return
    
    # Apply masks to datasets
    forget_dataset.data = forget_dataset.data[forget_mask]
    forget_dataset.targets = (-targets[forget_mask] - 1).tolist()  # Convert back to original labels
    
    retain_dataset.data = retain_dataset.data[retain_mask]
    retain_dataset.targets = targets[retain_mask].tolist()
    
    # Create data loaders (handle empty datasets)
    forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
    retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

    # Check if we have the required data for the method
    if args.unlearn == "GA" and (forget_loader is None or len(forget_dataset) == 0):
        print("❌ ERROR: No forget samples for GA method")
        return
    if args.unlearn in ["FT", "RL"] and (retain_loader is None or len(retain_dataset) == 0):
        print("❌ ERROR: No retain samples for FT/RL method")
        return

    # Verify split
    total_expected = len(train_loader_full.dataset)
    total_actual = len(forget_dataset) + len(retain_dataset)
    
    if total_actual != total_expected:
        print(f"⚠️  Dataset split warning: {total_actual} != {total_expected}")
    else:
        print(f"✅ Dataset split verified: {total_actual} = {total_expected}")

    print(f"📁 Dataset Info:")
    print(f"   Retain dataset: {len(retain_dataset)} samples")
    print(f"   Forget dataset: {len(forget_dataset)} samples")
    print(f"   Forget percentage: {forget_percentage*100:.0f}%")
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model file not found: {args.model_path}")
        print("You need to train a model first using main_train.py")
        print("Example training command:")
        print("python main_train.py --arch resnet18 --dataset cifar100 --lr 0.05 --epochs 200 --weight_decay 0.0005 --momentum 0.9 --save_dir saved_models --amp --batch_size 512 --workers 8")
        return

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        print(f"📥 Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Model loaded successfully!")

    save_gradient_ratio(unlearn_data_loaders, model, criterion, args)


if __name__ == "__main__":
    main()