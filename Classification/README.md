# AOS: Image Classification

This directory contains the code for applying Adaptive Otsu Unlearning (AOS) on Image Classification models such as ResNet and VGG.

## Requirements
Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Train the Original Model
Train a baseline classification model before applying unlearning:
```bash
python main_train.py --arch resnet18 --dataset cifar100 --lr 0.1 --epochs 100 --save_dir ./weights
```

### 2. Generate AOS Masks
Generate the adaptive masks based on Otsu thresholding:
```bash
python run_otsu_experiments.py --method FT --forget_ratio 0.1
```

### 3. Run Unlearning
Apply the unlearning algorithm using the generated masks:
```bash
python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --num_indexes_to_replace 4500 --model_path ./weights/model.pt --save_dir ./unlearned_weights --mask_path ./masks/mask_otsu.pt
```

## Evaluated Methods
- **RL (Retain Loss)**: Minimizes loss on the retain set while scaling gradients dynamically.
- **FT (Fine-tuning)**: Differentiates gradients between retain and forget sets.
- **GA (Gradient Ascent)**: Performs gradient ascent on the forget set to erase specific knowledge.

## Core Logic
The AOS methodology is primarily implemented in `otsu_utils.py`, handling bounded Otsu thresholding, variance stability, and retention-aware scaling.