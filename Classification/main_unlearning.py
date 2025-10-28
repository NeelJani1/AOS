import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import argparse
import copy
from collections import OrderedDict

# Import your existing modules
from utils import setup_model_dataset, test_model_accuracy, save_checkpoint
from utils import mark_dataset_for_percentage, split_marked_dataset, replace_loader_dataset

def setup_experiment_paths(args):
    """Setup organized directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_base_dir, f"{args.experiment_name}_{timestamp}")
    
    paths = {
        'experiment_dir': experiment_dir,
        'unlearning_dir': os.path.join(experiment_dir, 'unlearning'),
        'results_dir': os.path.join(experiment_dir, 'results'),
        'mia_dir': os.path.join(experiment_dir, 'mia_evaluation'),
        'logs_dir': os.path.join(experiment_dir, 'logs')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def load_model_and_mask(model_path, mask_path, device):
    """Load model and mask with error handling"""
    # Load model
    model, _, _, _, _ = setup_model_dataset(args)
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ Model loaded from: {model_path}")
    
    # Load mask
    mask = torch.load(mask_path, map_location=device)
    print(f"✅ Mask loaded from: {mask_path}")
    print(f"📊 Mask retention: {calculate_mask_retention(mask):.1f}%")
    
    return model, mask

def calculate_mask_retention(mask):
    """Calculate mask retention percentage"""
    total_params = sum(m.numel() for m in mask.values())
    retained_params = sum(m.sum().item() for m in mask.values())
    return (retained_params / total_params) * 100 if total_params > 0 else 0

class AdaptiveUnlearner:
    def __init__(self, model, mask, args, device):
        self.model = model
        self.mask = mask
        self.args = args
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Store initial state for frozen parameters
        self.initial_state = {}
        for name, param in model.named_parameters():
            if name in mask:
                self.initial_state[name] = param.data.clone()
    
    def apply_mask_to_gradients(self):
        """Apply mask to gradients during training"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask and param.grad is not None:
                    param.grad *= self.mask[name]
    
    def restore_frozen_parameters(self):
        """Restore initial state for frozen parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask and name in self.initial_state:
                    param.data = param.data * self.mask[name] + self.initial_state[name] * (1 - self.mask[name])
    
    def fine_tuning_unlearn(self, retain_loader, forget_loader):
        """Fine-Tuning approach: Train on retain set only"""
        print("🔧 Using Fine-Tuning (FT) unlearning...")
        
        optimizer = optim.SGD(
            self.model.parameters(),
            self.args.unlearn_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        
        best_accuracy = 0
        training_log = []
        
        for epoch in range(self.args.unlearn_epochs):
            self.model.train()
            train_loss = 0
            batch_idx = 0
            
            for batch_idx, (images, targets) in enumerate(retain_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                self.apply_mask_to_gradients()
                optimizer.step()
                self.restore_frozen_parameters()
                
                train_loss += loss.item()
            
            # Evaluate
            test_acc, retain_acc, forget_acc = self.evaluate_model(retain_loader, forget_loader)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            training_log.append({
                'epoch': epoch,
                'train_loss': train_loss / (batch_idx + 1),
                'test_accuracy': test_acc,
                'retain_accuracy': retain_acc,
                'forget_accuracy': forget_acc
            })
            
            if epoch % 2 == 0:
                print(f'Epoch {epoch}: Loss: {train_loss/(batch_idx+1):.3f}, '
                      f'Test: {test_acc:.2f}%, Retain: {retain_acc:.2f}%, Forget: {forget_acc:.2f}%')
        
        # Load best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        return training_log
    
    def gradient_ascent_unlearn(self, retain_loader, forget_loader):
        """Gradient Ascent approach: Controlled unlearning on forget set"""
        print("🔧 Using Gradient Ascent (GA) unlearning...")
        
        optimizer = optim.SGD(
            self.model.parameters(),
            self.args.unlearn_lr * 0.5,  # Lower LR for GA
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        
        training_log = []
        
        for epoch in range(self.args.unlearn_epochs):
            self.model.train()
            train_loss = 0
            batch_idx = 0
            
            for batch_idx, (images, targets) in enumerate(forget_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = -self.criterion(outputs, targets) * 0.1  # Conservative negative scaling
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.apply_mask_to_gradients()
                optimizer.step()
                self.restore_frozen_parameters()
                
                train_loss += loss.item()
                
                # Early stop to prevent catastrophic forgetting
                if batch_idx >= 10:
                    break
            
            # Evaluate
            test_acc, retain_acc, forget_acc = self.evaluate_model(retain_loader, forget_loader)
            
            training_log.append({
                'epoch': epoch,
                'train_loss': train_loss / (batch_idx + 1),
                'test_accuracy': test_acc,
                'retain_accuracy': retain_acc,
                'forget_accuracy': forget_acc
            })
            
            print(f'Epoch {epoch}: Loss: {train_loss/(batch_idx+1):.3f}, '
                  f'Test: {test_acc:.2f}%, Retain: {retain_acc:.2f}%, Forget: {forget_acc:.2f}%')
            
            # Early stopping if model is destroyed
            if retain_acc < 20.0:
                print("⚠️  Early stopping: Model retention too low")
                break
        
        return training_log
    
    def retain_learning_unlearn(self, retain_loader, forget_loader):
        """Retain Learning approach: Balance retaining and forgetting"""
        print("🔧 Using Retain Learning (RL) unlearning...")
        
        optimizer = optim.SGD(
            self.model.parameters(),
            self.args.unlearn_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        
        best_accuracy = 0
        training_log = []
        
        for epoch in range(self.args.unlearn_epochs):
            self.model.train()
            train_loss = 0
            batch_idx = 0
            
            # Phase 1: Retain learning
            for batch_idx, (images, targets) in enumerate(retain_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                self.apply_mask_to_gradients()
                optimizer.step()
                self.restore_frozen_parameters()
                
                train_loss += loss.item()
            
            # Evaluate
            test_acc, retain_acc, forget_acc = self.evaluate_model(retain_loader, forget_loader)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            training_log.append({
                'epoch': epoch,
                'train_loss': train_loss / (batch_idx + 1),
                'test_accuracy': test_acc,
                'retain_accuracy': retain_acc,
                'forget_accuracy': forget_acc
            })
            
            if epoch % 2 == 0:
                print(f'Epoch {epoch}: Loss: {train_loss/(batch_idx+1):.3f}, '
                      f'Test: {test_acc:.2f}%, Retain: {retain_acc:.2f}%, Forget: {forget_acc:.2f}%')
        
        # Load best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        return training_log
    
    def evaluate_model(self, retain_loader, forget_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        # Test accuracy (on validation set)
        test_acc = test_model_accuracy(self.model, self.test_loader, self.device)
        
        # Retain accuracy
        retain_acc = test_model_accuracy(self.model, retain_loader, self.device) if retain_loader else 0
        
        # Forget accuracy  
        forget_acc = test_model_accuracy(self.model, forget_loader, self.device) if forget_loader else 0
        
        return test_acc, retain_acc, forget_acc
    
    def set_test_loader(self, test_loader):
        """Set test loader for evaluation"""
        self.test_loader = test_loader

def run_unlearning_experiment(args, method, percentage, model_path, mask_path, paths):
    """Run unlearning for a specific method and percentage"""
    print(f"\n🎯 Starting {method} unlearning for {percentage*100:.0f}% forgetting...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and mask
    model, mask = load_model_and_mask(model_path, mask_path, device)
    
    # Prepare datasets
    original_model, train_loader_full, val_loader, _, _ = setup_model_dataset(args)
    original_dataset = copy.deepcopy(train_loader_full.dataset)
    
    # Mark and split dataset
    marked_dataset = mark_dataset_for_percentage(original_dataset, percentage, seed=args.seed)
    forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
    
    retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=args.seed, shuffle=True)
    forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=args.seed, shuffle=True)
    
    print(f"📊 Dataset: {len(retain_dataset)} retain, {len(forget_dataset)} forget samples")
    
    # Initialize unlearner
    unlearner = AdaptiveUnlearner(model, mask, args, device)
    unlearner.set_test_loader(val_loader)
    
    # Run unlearning based on method
    if method == 'FT':
        training_log = unlearner.fine_tuning_unlearn(retain_loader, forget_loader)
    elif method == 'GA':
        training_log = unlearner.gradient_ascent_unlearn(retain_loader, forget_loader)
    elif method == 'RL':
        training_log = unlearner.retain_learning_unlearn(retain_loader, forget_loader)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Final evaluation
    final_test_acc, final_retain_acc, final_forget_acc = unlearner.evaluate_model(retain_loader, forget_loader)
    
    # Save results
    results = {
        'method': method,
        'forget_percentage': percentage,
        'final_test_accuracy': final_test_acc,
        'final_retain_accuracy': final_retain_acc,
        'final_forget_accuracy': final_forget_acc,
        'training_log': training_log,
        'unlearn_epochs': args.unlearn_epochs,
        'mask_retention': calculate_mask_retention(mask),
        'retain_samples': len(retain_dataset),
        'forget_samples': len(forget_dataset)
    }
    
    # Save model and results
    output_dir = os.path.join(paths['unlearning_dir'], method, f"{int(percentage*100)}percent")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save({
        'state_dict': unlearner.model.state_dict(),
        'accuracy': final_test_acc,
        'epoch': args.unlearn_epochs,
    }, os.path.join(output_dir, 'model_best.pth.tar'))
    
    # Save results
    with open(os.path.join(output_dir, 'unlearning_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ {method} at {percentage*100:.0f}%: "
          f"Test={final_test_acc:.2f}%, Retain={final_retain_acc:.2f}%, Forget={final_forget_acc:.2f}%")
    
    return results, unlearner.model

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Unlearning Framework')
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--experiment_name', default='comprehensive_unlearning', type=str)
    parser.add_argument('--output_base_dir', default='results', type=str)
    parser.add_argument('--unlearn_epochs', default=10, type=int)
    parser.add_argument('--unlearn_lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--workers', default=8, type=int)
    
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_experiment_paths(args)
    
    # Define experiments to run
    experiments = [
        # Format: (method, percentage, mask_path_pattern)
        ('FT', 0.1, 'masks/FT/10percent/mask_otsu_FT_10percent_conservative.pt'),
        ('FT', 0.2, 'masks/FT/20percent/mask_otsu_FT_20percent_conservative.pt'),
        ('FT', 0.3, 'masks/FT/30percent/mask_otsu_FT_30percent_conservative.pt'),
        ('FT', 0.4, 'masks/FT/40percent/mask_otsu_FT_40percent_conservative.pt'),
        ('FT', 0.5, 'masks/FT/50percent/mask_otsu_FT_50percent_conservative.pt'),
        ('RL', 0.1, 'masks/RL/10percent/mask_otsu_RL_10percent_conservative.pt'),
        ('RL', 0.2, 'masks/RL/20percent/mask_otsu_RL_20percent_conservative.pt'),
        ('RL', 0.3, 'masks/RL/30percent/mask_otsu_RL_30percent_conservative.pt'),
        ('RL', 0.4, 'masks/RL/40percent/mask_otsu_RL_40percent_conservative.pt'),
        ('RL', 0.5, 'masks/RL/50percent/mask_otsu_RL_50percent_conservative.pt'),
    ]
    
    all_results = {}
    
    for method, percentage, mask_pattern in experiments:
        # Construct full mask path
        mask_path = os.path.join(paths['experiment_dir'], mask_pattern)
        
        if not os.path.exists(mask_path):
            print(f"❌ Mask not found: {mask_path}")
            continue
        
        try:
            results, model = run_unlearning_experiment(
                args, method, percentage, args.model_path, mask_path, paths
            )
            all_results[f"{method}_{int(percentage*100)}"] = results
        except Exception as e:
            print(f"❌ Error in {method} {percentage*100:.0f}%: {e}")
            continue
    
    # Save comprehensive results
    summary_path = os.path.join(paths['results_dir'], 'unlearning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 UNLEARNING COMPREHENSIVE SUMMARY")
    print("="*60)
    print("Method | Percentage | Test Acc | Retain Acc | Forget Acc")
    print("-" * 60)
    
    for key, res in all_results.items():
        print(f"{res['method']:6} | {res['forget_percentage']*100:>4.0f}%      | {res['final_test_accuracy']:>8.2f}% | "
              f"{res['final_retain_accuracy']:>10.2f}% | {res['final_forget_accuracy']:>9.2f}%")
    
    print(f"\n💾 All results saved in: {paths['experiment_dir']}")

if __name__ == '__main__':
    main()