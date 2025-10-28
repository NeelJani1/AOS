import copy
import os
import time

import torch
import utils
from imagenet import get_x_y_from_data_dict
import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False, use_amp=False, scaler=None):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    # === START OF OLD FIX ===
    # We must manually define the epoch length for streaming ImageNet
    if args.dataset == "imagenet":
        # ImageNet has 1,281,167 training images
        one_epoch_step = 1281167 // args.batch_size
    else:
        one_epoch_step = len(train_loader)
    # === END OF OLD FIX ===

    start = time.time()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    for i, data in enumerate(train_loader):
        
        # === START OF NEW FIX ===
        # Handle different data loader formats
        if args.dataset == "imagenet":
            # Streaming ImageNet returns a dict
            image, target = get_x_y_from_data_dict(data, device)
        else:
            # TinyImagenet/CIFAR return a list [images, labels]
            image, target = data
            image = image.to(device)
            target = target.to(device)

        if image is None: # Safety check
            print(f"Error: Data unpacking failed for dataset {args.dataset}. Skipping batch {i}.")
            continue
        # === END OF NEW FIX ===
        
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=one_epoch_step, args=args
            )
        
        # compute output
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
            
            scaler.scale(loss).backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            scaler.step(optimizer)
            scaler.update()
        else:
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, one_epoch_step, end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg


def validate(val_loader, model, criterion, args, use_amp=False, scaler=None):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Handle different data loader formats (same as training)
            if args.dataset == "imagenet":
                # Streaming ImageNet returns a dict
                image, target = get_x_y_from_data_dict(data, device)
            else:
                # TinyImagenet/CIFAR return a list [images, labels]
                image, target = data
                image = image.to(device)
                target = target.to(device)

            if image is None: # Safety check
                print(f"Error: Data unpacking failed for dataset {args.dataset}. Skipping batch {i}.")
                continue

            # compute output
            if use_amp:
                with autocast():
                    output = model(image)
                    loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg