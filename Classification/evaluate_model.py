import argparse
import os
import copy
import numpy as np
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ---
# This script combines all necessary code from:
# - arg_parser.py (for default args)
# - utils.py (for models, datasets, and helpers)
# - trainer.py (for the validate function)
# - generate_masks.py (for the data splitting logic)
# ---

#
# --- From arg_parser.py ---
#
def parse_args_defaults():
    """
    Parses the default arguments from your arg_parser.py
    """
    parser = argparse.ArgumentParser(description='PyTorch Training')
    
    # --- We only care about the defaults, so we parse an empty list ---
    # (This is a trick to get the default values)
    
    # Basic parameters
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet'])
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50', 'mobilenet_v2', 'vgg16_bn'])
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--workers', default=8, type=int)
    
    # Training parameters
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--print_freq', default=50, type=int)
    
    # Unlearning parameters
    parser.add_argument('--unlearn_lr', default=0.1, type=float)
    
    # Other parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--amp', action='store_true', default=True) # Defaulting to True as in your train script

    # We parse an empty list to just get the defaults
    args, _ = parser.parse_known_args([]) 
    return args

#
# --- From utils.py ---
#
def setup_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model(args):
    if args.arch == "resnet18":
        model = resnet18_cifar(num_classes=args.num_classes)
    elif args.arch == "resnet50":
        model = resnet50_cifar(num_classes=args.num_classes)
    elif args.arch == "mobilenet_v2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(num_classes=args.num_classes)
    elif args.arch == "vgg16_bn":
        from torchvision.models import vgg16_bn
        model = vgg16_bn(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    return model

def resnet18_cifar(num_classes=100):
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def resnet50_cifar(num_classes=100):
    from torchvision.models import resnet50
    model = resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_dataset(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        return get_cifar10(args)
    elif args.dataset == "cifar100":
        args.num_classes = 100
        return get_cifar100(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")

def get_cifar10(args):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, val_loader # train, val, test

def get_cifar100(args):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, val_loader # train, val, test

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#
# --- From trainer.py ---
#
def validate(val_loader, model, criterion, args, use_amp=False, scaler=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # ---
            # FIX: Added data unpacking logic from your `train` function
            # ---
            if args.dataset == "imagenet":
                # This logic was in your trainer.py but not used.
                # Keeping it for completeness.
                image, target = get_x_y_from_data_dict(data, device)
            else:
                # Standard CIFAR/TinyImagenet
                image, target = data
                image = image.to(device)
                target = target.to(device)

            if image is None: 
                print(f"Error: Data unpacking failed. Skipping batch {i}.")
                continue
            # --- End of FIX ---
            
            if use_amp:
                with autocast():
                    output = model(image)
                    loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

    # Don't print stats, just return the average
    return top1.avg

#
# --- From generate_masks.py (Data Splitting Logic) ---
#
def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
    np.random.seed(seed)
    try:
        targets_np = np.array(dataset.targets)
    except:
        targets_np = np.array([s[1] for s in dataset.samples])
        
    total_samples = len(targets_np)
    num_to_forget = int(forget_percentage * total_samples)
    if num_to_forget > total_samples:
        num_to_forget = total_samples
        
    forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False)
    
    if hasattr(dataset, 'targets'):
        if not isinstance(dataset.targets, list):
             dataset.targets = list(dataset.targets)
        for idx in forget_indices:
            if dataset.targets[idx] >= 0:
                dataset.targets[idx] = -dataset.targets[idx] - 1
    elif hasattr(dataset, 'samples'):
        for idx in forget_indices:
            if dataset.samples[idx][1] >= 0:
                dataset.samples[idx] = (dataset.samples[idx][0], -dataset.samples[idx][1] - 1)
    else:
        raise ValueError("Unknown dataset structure. Cannot mark targets.")
    return dataset

def split_marked_dataset(marked_dataset):
    forget_dataset = copy.deepcopy(marked_dataset)
    retain_dataset = copy.deepcopy(marked_dataset)
    
    if hasattr(marked_dataset, 'data') and hasattr(marked_dataset, 'targets'):
        targets = np.array(marked_dataset.targets)
        forget_mask = targets < 0
        retain_mask = targets >= 0 
        
        forget_dataset.data = forget_dataset.data[forget_mask, ...]
        forget_dataset.targets = (-targets[forget_mask] - 1).tolist()
        
        retain_dataset.data = retain_dataset.data[retain_mask, ...]
        retain_dataset.targets = targets[retain_mask].tolist()

    elif hasattr(marked_dataset, 'samples'):
        all_samples = marked_dataset.samples
        forget_samples = []
        retain_samples = []
        for (path, target) in all_samples:
            if target < 0:
                forget_samples.append((path, -target - 1))
            else:
                retain_samples.append((path, target))
                
        forget_dataset.samples = forget_samples
        forget_dataset.imgs = forget_samples 
        retain_dataset.samples = retain_samples
        retain_dataset.imgs = retain_samples
        
        if hasattr(forget_dataset, 'targets'):
            forget_dataset.targets = [s[1] for s in forget_samples]
        if hasattr(retain_dataset, 'targets'):
            retain_dataset.targets = [s[1] for s in retain_samples]
    else:
        raise ValueError("Unknown dataset structure. Cannot split.")
    return forget_dataset, retain_dataset

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True, args=None):
    if dataset is None or len(dataset) == 0:
        return None
    
    setup_seed(seed) # Reset seed for consistent shuffling
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.workers if args else 8, 
        pin_memory=True,
        shuffle=shuffle,
    )

#
# --- Main Evaluation Function ---
#
def main():
    # 1. Parse arguments for this script
    parser = argparse.ArgumentParser(description='Evaluate a single model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model_SA_best.pth.tar file')
    parser.add_argument('--forget_perc', type=float, required=True,
                        help='The forgetting percentage (e.g., 0.1, 0.2)')
    eval_args = parser.parse_args()

    # 2. Load all *default* arguments from your project
    args = parse_args_defaults()
    
    # 3. Update defaults with the specific args for this run
    args.model_path = eval_args.model_path
    args.forget_percentage = eval_args.forget_perc
    
    # 4. Setup device and seed
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        args.device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        args.device = torch.device("cpu")
    setup_seed(args.seed)

    # 5. Load full dataset and define model
    # (This gets the full 50k train set and 10k test set)
    train_loader_full, val_loader, test_loader = get_dataset(args)
    
    # Set num_classes based on dataset
    if args.dataset == 'cifar10': args.num_classes = 10
    if args.dataset == 'cifar100': args.num_classes = 100
    
    model = get_model(args).to(args.device)

    # 6. Load the saved model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
        
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Load state_dict (using logic from your main_train.py)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        # Fallback if it's just the state dict
        model.load_state_dict(checkpoint, strict=True)
        
    model.eval()
    
    # 7. Create the Forget, Retain, and Test dataloaders
    
    # 7a. Get the full (50k) training dataset object
    original_dataset = copy.deepcopy(train_loader_full.dataset)
    
    # 7b. Mark it
    marked_dataset = mark_dataset_for_percentage(
        original_dataset, args.forget_percentage, seed=args.seed
    )
    
    # 7c. Split it
    forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
    
    # 7d. Create the DataLoaders
    forget_loader = replace_loader_dataset(
        forget_dataset, args.batch_size, seed=args.seed, shuffle=False, args=args
    )
    retain_loader = replace_loader_dataset(
        retain_dataset, args.batch_size, seed=args.seed, shuffle=False, args=args
    )
    
    # The 'val_loader' from get_dataset is our Test Set
    test_loader = val_loader 

    # 8. Run Evaluations
    criterion = nn.CrossEntropyLoss()
    
    # Handle empty loaders (e.g., 100% forget)
    forget_acc = 0.0
    if forget_loader is not None:
        forget_acc = validate(forget_loader, model, criterion, args, use_amp=args.amp)
        
    retain_acc = 0.0
    if retain_loader is not None:
        retain_acc = validate(retain_loader, model, criterion, args, use_amp=args.amp)
        
    test_acc = 0.0
    if test_loader is not None:
        test_acc = validate(test_loader, model, criterion, args, use_amp=args.amp)

    # 9. Print results as a simple CSV string
    # This format is easy for the bash script to capture.
    print(f"{forget_acc:.2f},{retain_acc:.2f},{test_acc:.2f}")

if __name__ == '__main__':
    main()