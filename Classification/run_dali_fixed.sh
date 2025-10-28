#!/bin/bash
cd /home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification
taskset -c 0-15 python main_train.py --arch resnet18 --dataset imagenet --epochs 80 --lr 0.1 --save_dir /home/neel/Unlearn-Saliency-master/saved_models_1 --batch_size 192 --workers 3
