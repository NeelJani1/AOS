#!/bin/bash
python main_train.py --arch resnet18 --dataset imagenet --epochs 80 --lr 0.1 --save_dir /home/neel/Unlearn-Saliency-master/saved_models_1 --batch_size 192 --workers 3
