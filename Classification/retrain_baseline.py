# retrain_baseline.py - WITH EARLY STOPPING
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import sys
import copy
import argparse
import numpy as np
from datetime import datetime

sys.path.append('.')
from utils import setup_model_dataset, save_checkpoint, mark_dataset_for_percentage, split_marked_dataset, replace_loader_dataset

def test_model_accuracy(model, test_loader, device):
    """Test model accuracy"""
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
    return 100 * correct / total if total > 0 else 0

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_accuracy = 0
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False
        
    def __call__(self, current_accuracy, epoch, model):
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            print(f"🎯 New best accuracy: {current_accuracy:.2f}% (epoch {epoch})")
            return False
        else:
            self.counter += 1
            print(f"⏳ No improvement: {self.counter}/{self.patience} (best: {self.best_accuracy:.2f}%)")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                if self.restore_best:
                    print(f"🔄 Restoring best model from epoch {self.best_epoch}")
                    model.load_state_dict(self.best_model_state)
                return True
            return False

def retrain_from_scratch(args, forget_percentage, device='cuda'):
    """
    Gold standard baseline with early stopping
    """
    print(f"\n🚀 RETRAIN BASELINE: Training from scratch for {forget_percentage*100:.0f}% forgetting...")
    
    # Get retain dataset
    model, train_loader_full, val_loader, test_loader, _ = setup_model_dataset(args)
    original_dataset = copy.deepcopy(train_loader_full.dataset)
    marked_dataset = mark_dataset_for_percentage(original_dataset, forget_percentage, seed=args.seed)
    forget_dataset, retain_dataset = split_marked_dataset(marked_dataset)
    
    if retain_dataset is None or len(retain_dataset) == 0:
        print(f"❌ No retain samples for {forget_percentage*100:.0f}% forgetting")
        return None
    
    print(f"📊 Retain dataset: {len(retain_dataset)} samples")
    
    # Create data loader for retain set
    retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=args.seed, shuffle=True)
    
    # Create fresh model
    model, _, _, _, _ = setup_model_dataset(args)
    model = model.to(device)
    
    # Training setup
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.05,
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.retrain_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,  # Default 10 epochs
        min_delta=0.1,  # 0.1% minimum improvement
        restore_best=True
    )
    
    # === NEW ===
    # Define output directory *before* the loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/baselines/retrain_{int(forget_percentage*100)}percent_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Saving results to: {output_dir}")
    # ===========

    # Training loop
    training_log = []
    best_accuracy = 0
    
    print(f"🔄 Training for maximum {args.retrain_epochs} epochs (early stopping patience: {args.patience})...")
    
    for epoch in range(args.retrain_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        
        for batch_idx, (data, target) in enumerate(retain_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        # Evaluate
        test_acc = test_model_accuracy(model, val_loader, device)
        retain_acc = test_model_accuracy(model, retain_loader, device)
        
        # Update best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        
        training_log.append({
            'epoch': epoch,
            'train_loss': train_loss/(batch_idx+1),
            'test_accuracy': test_acc,
            'retain_accuracy': retain_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Print progress every 5 epochs or first/last epochs
        if epoch % 5 == 0 or epoch < 5 or epoch == args.retrain_epochs - 1:
            print(f'  Epoch {epoch:3d}: Loss: {train_loss/(batch_idx+1):.3f}, '
                  f'Test Acc: {test_acc:.2f}%, Retain Acc: {retain_acc:.2f}%, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # === CHANGED ===
        # Check early stopping
        stop_training = early_stopping(test_acc, epoch, model)
        
        # Save checkpoint if this is the new best model
        # (The counter is reset to 0 inside EarlyStopping when a new best is found)
        if early_stopping.counter == 0: 
            print(f"💾 Saving new best model to {output_dir}")
            torch.save({
                'state_dict': early_stopping.best_model_state, # Use the state saved by the stopper
                'accuracy': early_stopping.best_accuracy,
                'best_epoch': early_stopping.best_epoch,
                'total_epochs': epoch + 1,
                'early_stopping_triggered': False # Not yet
            }, os.path.join(output_dir, 'model_best.pth.tar')) # This will overwrite

        if stop_training:
            print(f"✅ Early stopping at epoch {epoch}")
            
            # Update the saved file one last time to mark it as stopped
            torch.save({
                'state_dict': early_stopping.best_model_state, 
                'accuracy': early_stopping.best_accuracy,
                'best_epoch': early_stopping.best_epoch,
                'total_epochs': epoch + 1,
                'early_stopping_triggered': True # Mark as stopped
            }, os.path.join(output_dir, 'model_best.pth.tar'))
            
            break # Exit the loop
        # ===============
    
    # Final evaluation with best model (already restored by early stopping)
    final_test_acc = test_model_accuracy(model, val_loader, device)
    final_retain_acc = test_model_accuracy(model, retain_loader, device)
    
    # For retrain baseline, forget accuracy should be random
    num_classes = 100  # CIFAR-100
    random_accuracy = 100.0 / num_classes
    
    # Calculate actual training efficiency
    efficiency = (early_stopping.best_epoch + 1) / args.retrain_epochs * 100
    print(f"📈 Training efficiency: {efficiency:.1f}% (stopped at epoch {early_stopping.best_epoch + 1})")
    
    # Save results
    results = {
        'forget_percentage': forget_percentage,
        'method': 'retrain_baseline',
        'final_test_accuracy': final_test_acc,
        'final_retain_accuracy': final_retain_acc,
        'final_forget_accuracy': random_accuracy,
        'best_test_accuracy': early_stopping.best_accuracy,
        'best_epoch': early_stopping.best_epoch,
        'total_epochs_trained': epoch + 1,
        'early_stopping_triggered': early_stopping.early_stop,
        'training_efficiency': efficiency,
        'retrain_epochs': args.retrain_epochs,
        'patience': args.patience,
        'retain_samples': len(retain_dataset),
        'training_log': training_log
    }
    
    # === CHANGED ===
    # The output_dir is already created. We just save the results.json here.
    # The model saving block that was here has been REMOVED.
    # =================
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Retrain baseline completed!")
    print(f"   Best Test Acc: {early_stopping.best_accuracy:.2f}% (epoch {early_stopping.best_epoch})")
    print(f"   Final Test Acc: {final_test_acc:.2f}%")
    print(f"   Retain Acc: {final_retain_acc:.2f}%")
    print(f"   Expected Forget Acc: {random_accuracy:.2f}% (random)")
    
    return results

# Add this evaluation to your retrain_baseline.py
def evaluate_forget_accuracy(model, forget_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Retrain from scratch baseline with early stopping')
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', default=1024, type=int)  # Increased batch size
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--percentages', nargs='+', type=float, 
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help='Forgetting percentages')
    parser.add_argument('--retrain_epochs', default=100, type=int, help='Maximum epochs for retraining')
    parser.add_argument('--patience', default=15, type=int, help='Early stopping patience')
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--workers', default=8, type=int)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    print(f"⚡ Early stopping: {args.patience} epochs patience")
    
    all_results = []
    for percent in args.percentages:
        try:
            results = retrain_from_scratch(args, percent, device)
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"❌ Error for {percent*100:.0f}%: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("📊 RETRAIN BASELINE SUMMARY WITH EARLY STOPPING")
    print("="*70)
    print("Percentage | Best Test Acc | Retain Acc | Epochs | Efficiency")
    print("-" * 70)
    for res in all_results:
        print(f"   {res['forget_percentage']*100:3.0f}%    |  {res['best_test_accuracy']:6.2f}%     |  {res['final_retain_accuracy']:6.2f}%   |  {res['best_epoch']+1:3d}/{res['total_epochs_trained']:3d} |     {res['training_efficiency']:5.1f}%")
    
    # Save comprehensive summary
    summary_dir = "results/baselines"
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'retrain_baseline_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main()