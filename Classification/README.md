# AOS: Image Classification

This directory contains the implementation of **Adaptive Otsu Unlearning (AOS)** for deep convolutional classification models (ResNet-18, VGG) evaluated on CIFAR-100.

---

## 🛠️ Setup & Requirements

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Step 1: Train Baseline Model
Train a standard classification model on CIFAR-100 before unlearning:
```bash
python main_train.py --arch resnet18 --dataset cifar100 --epochs 100 --lr 0.1 --save_dir ./weights
```

### Step 2: Execute AOS Unlearning
Run the full adaptive Otsu unlearning pipeline across multiple forget ratios and unlearning regimes (RL, FT, GA):
```bash
python main_unlearning.py --arch resnet18 --dataset cifar100 --model_path ./weights/resnet18_cifar100.pth
```

Or test specific Otsu thresholding configurations and methods:
```bash
python run_otsu_experiments.py --test_methods
# Or run all automated parameter sweeps:
python run_otsu_experiments.py --run_all
```

### Step 3: Evaluation & Privacy Auditing
Evaluate test accuracy, retain accuracy, and forget accuracy:
```bash
python evaluate_model.py --model_path ./results/unlearned_model.pt --forget_perc 0.1
```

Run Membership Inference Attack (MIA) verification to audit concept erasure:
```bash
python evaluate_all_mia.py --dataset cifar100 --arch resnet18
```

---

## 🔬 Core Components & Logic

- **`otsu_utils.py`**: Core mathematical engine implementing:
  - `enhanced_otsu_threshold()`: Maximizes between-class variance over saliency distributions.
  - `bounded_otsu()`: Enforces empirical retention bounds $[\tau_{\min}, \tau_{\max}]$.
  - `generate_otsu_mask()`: Computes Fisher-weighted, layer-wise sparse unlearning gates.
  - `compute_gradients()`: Saliency and curvature extraction across unlearning regimes.
- **`main_unlearning.py`**: End-to-end unlearning loop with dynamic annealing and retention scaling.
- **`run_otsu_experiments.py`**: Automated test harnesses for systematic benchmark reproduction.