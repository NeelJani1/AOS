train_code = '''import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler

import utils


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    
    # === DALI Integration ===
    is_imagenet_streaming = args.dataset == "imagenet" # This is now our DALI flag
    
    if is_imagenet_streaming:
        # Create iterator from DataLoader
        train_loader_iter = iter(train_loader)
        i = 0
        while True:
            try:
                # measure data loading time
                data_time.update(time.time() - end)

                # Get next batch from DALI iterator
                data = next(train_loader_iter)
                
                # Extract images and labels
                # Data is already on GPU
                image = data[0]['images']
                target = data[0]['labels'].squeeze(-1).long() # Remove extra dim and cast to long

                # compute output
                output = model(image)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1 = utils.accuracy(output.data, target)[0]
                losses.update(loss.item(), image.size(0))
                top1.update(prec1.item(), image.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                            epoch,
                            i,
                            args.one_epoch_step, # Use the pre-calculated step
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1,
                        )
                    )
                i += 1 # Increment batch counter

            except StopIteration:
                # End of epoch
                break
    
    else:
        # Original loop for non-ImageNet datasets
        for i, (image, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            image = image.cuda()
            target = target.cuda()

            # compute output
            output = model(image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                    )
                )
    # === End DALI Integration ===

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg
'''

with open('trainer/train.py', 'w') as f:
    f.write(train_code)
    
print("trainer/train.py has been fixed!")
