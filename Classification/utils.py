import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

def setup_seed(seed):
    """
    Setup random seed for reproducibility
    """
    if seed is None:
        return
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_model_dataset(args):
    """
    Setup model and dataset based on arguments
    """
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000
    elif args.dataset == "tiny_imagenet":
        args.num_classes = 200
    else:
        raise ValueError("Unknown dataset")
    
    # Setup model
    model = get_model(args)
    
    # Setup datasets
    train_loader, val_loader, test_loader, marked_loader = get_dataset(args)
    
    return model, train_loader, val_loader, test_loader, marked_loader

def get_model(args):
    """
    Get model based on architecture - FIXED for CIFAR ResNet
    """
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

# CIFAR-compatible ResNet models
def resnet18_cifar(num_classes=100):
    """ResNet-18 model for CIFAR with 3x3 conv1"""
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    # Modify first convolution for CIFAR (3x3 kernel instead of 7x7)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the first maxpool layer for CIFAR
    model.maxpool = nn.Identity()
    return model

def resnet50_cifar(num_classes=100):
    """ResNet-50 model for CIFAR with 3x3 conv1"""
    from torchvision.models import resnet50
    model = resnet50(num_classes=num_classes)
    # Modify first convolution for CIFAR (3x3 kernel instead of 7x7)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the first maxpool layer for CIFAR
    model.maxpool = nn.Identity()
    return model

def get_dataset(args):
    """
    Get dataset and data loaders
    """
    if args.dataset == "cifar10":
        return get_cifar10(args)
    elif args.dataset == "cifar100":
        return get_cifar100(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")

def get_cifar10(args):
    """
    Get CIFAR-10 dataset
    """
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
    
    # For unlearning, we use the same dataset but will mark it later
    marked_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader, val_loader, marked_loader

def get_cifar100(args):
    """
    Get CIFAR-100 dataset
    """
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
    
    # For unlearning, we use the same dataset but will mark it later
    marked_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader, val_loader, marked_loader

def save_checkpoint(state, is_SA_best, save_path, pruning=None, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    
    if is_SA_best:
        best_filename = os.path.join(save_path, 'model_SA_best.pth.tar')
        torch.save(state, best_filename)

def count_parameters(model):
    """
    Count total parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model):
    """
    Print model information
    """
    total_params = count_parameters(model)
    print(f"📊 Model has {total_params:,} parameters")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   {name}: {param.numel():,} parameters")

class AverageMeter(object):
    """Computes and stores average and current value"""
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
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def test_model_accuracy(model, test_loader, device):
    """
    Test model accuracy on a dataset - COMPREHENSIVE VERSION
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def test(model, test_loader, device):
    """
    Alias for test_model_accuracy for compatibility
    """
    return test_model_accuracy(model, test_loader, device)

# ========== MISSING FUNCTIONS ADDED BELOW ==========

def mark_dataset_for_percentage(dataset, forget_percentage, seed=42):
    """
    Mark dataset for specific forgetting percentage
    """
    np.random.seed(seed)
    
    try:
        targets_np = np.array(dataset.targets)
    except:
        # For datasets that use samples instead of targets
        targets_np = np.array([s[1] for s in dataset.samples])
        
    total_samples = len(targets_np)
    num_to_forget = int(forget_percentage * total_samples)
    
    if num_to_forget > total_samples:
        num_to_forget = total_samples
        
    forget_indices = np.random.choice(total_samples, size=num_to_forget, replace=False)
    
    print(f"🎯 Marking {num_to_forget} samples ({forget_percentage*100:.0f}%) for forgetting")
    
    if hasattr(dataset, 'targets'):
        # Ensure targets is a list for mutation
        if not isinstance(dataset.targets, list):
            dataset.targets = dataset.targets.tolist() if hasattr(dataset.targets, 'tolist') else list(dataset.targets)
             
        for idx in forget_indices:
            if dataset.targets[idx] >= 0:
                dataset.targets[idx] = -dataset.targets[idx] - 1
        targets_after = np.array(dataset.targets)
    
    elif hasattr(dataset, 'samples'):
        for idx in forget_indices:
            if dataset.samples[idx][1] >= 0:
                dataset.samples[idx] = (dataset.samples[idx][0], -dataset.samples[idx][1] - 1)
        targets_after = np.array([s[1] for s in dataset.samples])

    else:
        print("❌ ERROR: Unknown dataset structure. Cannot mark targets.")
        return dataset

    forget_count = np.sum(targets_after < 0)
    retain_count = np.sum(targets_after >= 0)
    
    print(f"✅ Marking complete: {forget_count} forget samples, {retain_count} retain samples")
    
    return dataset

def split_marked_dataset(marked_dataset):
    """
    Splits a dataset marked with negative targets into forget and retain datasets.
    Handles both standard (CIFAR) and ImageFolder-style datasets.
    """
    forget_dataset = copy.deepcopy(marked_dataset)
    retain_dataset = copy.deepcopy(marked_dataset)
    
    # Case 1: Standard torchvision dataset (e.g., CIFAR10/100)
    if hasattr(marked_dataset, 'data') and hasattr(marked_dataset, 'targets'):
        targets = np.array(marked_dataset.targets)
        forget_mask = targets < 0
        retain_mask = targets >= 0 
        
        # Apply masks to forget dataset
        forget_dataset.data = marked_dataset.data[forget_mask, ...]
        forget_dataset.targets = (-targets[forget_mask] - 1).tolist()
        
        # Apply masks to retain dataset
        retain_dataset.data = marked_dataset.data[retain_mask, ...]
        retain_dataset.targets = targets[retain_mask].tolist()

    # Case 2: ImageFolder-style dataset
    elif hasattr(marked_dataset, 'samples'):
        all_samples = marked_dataset.samples
        forget_samples = []
        retain_samples = []
        
        for (path, target) in all_samples:
            if target < 0: # Forget sample
                forget_samples.append((path, -target - 1))
            else: # Retain sample
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
        print("❌ ERROR: Unknown dataset structure. Cannot split.")
        return None, None

    return forget_dataset, retain_dataset

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    """
    Creates a new DataLoader for a given dataset, handling empty datasets.
    """
    if dataset is None or len(dataset) == 0:
        return None
    
    setup_seed(seed) # Reset seed for consistent shuffling
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4, 
        pin_memory=True,
        shuffle=shuffle,
    )

def compute_gradients(model, data_loader, device, method='FT'):
    """
    Compute gradients for different unlearning methods
    """
    model.eval()
    gradients = {}
    
    # Initialize gradients dictionary
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.zeros_like(param)
    
    if method == 'FT':
        # Fine-tuning approach - standard training on retain set
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            model.zero_grad()
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] += param.grad.detach().clone()
            
            if batch_idx >= 5:  # Limit batches for efficiency
                break
                
    elif method == 'GA':
        # Gradient Ascent on forget set
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = -nn.CrossEntropyLoss()(output, target)  # Negative for ascent
            
            model.zero_grad()
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] += param.grad.detach().clone()
            
            if batch_idx >= 5:
                break
                
    # Clear any remaining gradients
    model.zero_grad()
    
    return gradients

def apply_mask_with_initial_state(model, mask, initial_state):
    """
    Apply mask while preserving initial state for frozen parameters
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask and name in initial_state:
                param.data = param.data * mask[name] + initial_state[name] * (1 - mask[name])

def calculate_mask_retention(mask):
    """
    Calculate what percentage of parameters are retained in mask
    """
    total_params = sum(m.numel() for m in mask.values())
    retained_params = sum(m.sum().item() for m in mask.values())
    return retained_params / total_params * 100 if total_params > 0 else 0

# Additional utility function for model loading
def load_model_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model checkpoint with proper error handling
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Model loaded successfully from: {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None