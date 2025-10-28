import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import arg_parser
from trainer import train, validate
from utils import setup_seed, setup_model_dataset, save_checkpoint

best_sa = 0

def debug_batch(batch, batch_idx=0):
    """Debug function to inspect batch contents"""
    print(f"Batch {batch_idx} type: {type(batch)}")
    if isinstance(batch, dict):
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
            if isinstance(value, torch.Tensor):
                print(f"    Tensor dtype: {value.dtype}, min: {value.min():.3f}, max: {value.max():.3f}")
    elif isinstance(batch, (list, tuple)):
        print(f"Batch length: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"  Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
    else:
        print(f"Batch attributes: {dir(batch)}")

def main():
    global best_sa
    args = arg_parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed:
        setup_seed(args.seed)

    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
        test_loader = None
        marked_loader = None
    else:
        model, train_loader, val_loader, test_loader, marked_loader = setup_model_dataset(args)

    model = model.cuda()

    if args.dataset == "imagenet":
        print("Number of train samples ~1.28M (streaming)")
        print("Number of val samples ~50K (streaming)")
    else:
        print(f"Number of train samples: {len(train_loader.dataset)}")
        print(f"Number of val samples: {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    milestones = list(map(int, args.decreasing_lr.split(",")))
    if args.imagenet_arch:
        def lr_lambda(cur_iter):
            if cur_iter < args.warmup:
                return (cur_iter + 1) / args.warmup
            else:
                progress = (cur_iter - args.warmup) / (args.epochs - args.warmup)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scaler = GradScaler()  # AMP gradient scaler

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=f"cuda:{args.gpu}")
        best_sa = checkpoint.get("best_sa", 0)
        start_epoch = checkpoint.get("epoch", 0)
        all_result = checkpoint.get("result", {"train_ta": [], "val_ta": [], "test_ta": []})

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))

        print(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_sa}")
    else:
        all_result = {"train_ta": [], "val_ta": [], "test_ta": []}

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch #{epoch}, Learning rate: {current_lr:.6f}")

        # Training loop with AMP
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, dict):
                # Hugging Face dataset format
                if 'pixel_values' in batch:
                    inputs = batch['pixel_values']
                elif 'image' in batch:
                    inputs = batch['image']
                else:
                    # Try to find the image tensor by checking all values
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 4:  # Batch x Channel x Height x Width
                            inputs = value
                            break
                    else:
                        print(f"Could not find image tensor in batch keys: {list(batch.keys())}")
                        continue
                
                if 'label' in batch:
                    targets = batch['label']
                elif 'labels' in batch:
                    targets = batch['labels']
                else:
                    # Try to find the label tensor
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 1:  # Batch of labels
                            targets = value
                            break
                    else:
                        print(f"Could not find label tensor in batch keys: {list(batch.keys())}")
                        continue
                        
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Standard tuple format (inputs, targets)
                inputs, targets = batch
            else:
                print(f"Unexpected batch format: {type(batch)}")
                if i == 0:  # Debug first unexpected batch
                    debug_batch(batch, i)
                continue

            # Debug print first batch only
            if i == 0:
                print(f"Batch {i} - Inputs: {inputs.shape}, Targets: {targets.shape}")
            
            # Check if we have tensors
            if not isinstance(inputs, torch.Tensor):
                print(f"Error: inputs is not a tensor, but {type(inputs)}. Skipping batch.")
                continue
            if not isinstance(targets, torch.Tensor):
                print(f"Error: targets is not a tensor, but {type(targets)}. Skipping batch.")
                continue

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            optimizer.zero_grad()

            with autocast():  # Enable mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total_samples += inputs.size(0)

            if (i + 1) % args.print_freq == 0:
                batch_loss = running_loss / total_samples if total_samples > 0 else 0
                batch_acc = running_corrects / total_samples if total_samples > 0 else 0
                print(f"Batch {i+1}/{len(train_loader)} - Loss: {batch_loss:.4f} - Acc: {batch_acc:.4f}")

        train_acc = running_corrects / total_samples if total_samples > 0 else 0.0

        # Validation
        print("Running validation...")
        val_acc = validate(val_loader, model, criterion, args, use_amp=True, scaler=scaler)

        scheduler.step()

        all_result["train_ta"].append(train_acc)
        all_result["val_ta"].append(val_acc)

        is_best = val_acc > best_sa
        best_sa = max(val_acc, best_sa)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            is_SA_best=is_best,
            pruning=0,
            save_path=args.save_dir,
        )

        print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_sa:.4f}")
        print(f"Epoch duration: {time.time() - start_time:.2f} seconds")

    plt.plot(all_result["train_ta"], label="Train Accuracy")
    plt.plot(all_result["val_ta"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "training_curve.png"))
    plt.close()

    print("Final evaluation on validation set:")
    final_acc = validate(val_loader, model, criterion, args, use_amp=True, scaler=scaler)

    if all_result["val_ta"]:
        best_epoch = np.argmax(all_result["val_ta"])
        print(f"* Best validation accuracy: {all_result['val_ta'][best_epoch]:.4f} at epoch {best_epoch + 1}")

if __name__ == "__main__":
    main()