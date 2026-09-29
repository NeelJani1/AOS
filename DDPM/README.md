# AOS: Generative Diffusion Unlearning (DDPM)

This directory contains the implementation of **Adaptive Otsu Saliency (AOS)** applied to conditional Denoising Diffusion Probabilistic Models (DDPM). AOS computes adaptive, layer-wise saliency thresholds to systematically unlearn specific generative classes while maintaining fidelity on retained concepts.

The framework builds upon and extends the architectures from [SalUn](https://github.com/optml-group/unlearn-saliency), [DDIM](https://github.com/ermongroup/ddim), and [Selective Amnesia (SA)](https://github.com/clear-nus/selective-amnesia).

---

## 🛠️ Requirements & Setup

Install the requirements using a `conda` environment:
```bash
conda create --name aos-ddpm python=3.10
conda activate aos-ddpm
pip install -r requirements.txt
```

---

## 🚀 Forgetting Training with AOS Saliency Unlearning

### Step 1: Train Conditional DDPM Baseline
Train a conditional DDPM on all 10 CIFAR-10 classes (specify GPU devices with `CUDA_VISIBLE_DEVICES`):

```bash
CUDA_VISIBLE_DEVICES="0,1" python train.py --config configs/cifar10_train.yml --mode train
```
The checkpoint will be saved under `results/cifar10/yyyy_mm_dd_hhmmss`.

### Step 2: Generate Adaptive Saliency Masks
Generate layer-wise adaptive saliency masks targeting the concept/class to forget (e.g., class `0` - airplane):

```bash
CUDA_VISIBLE_DEVICES="0,1" python train.py --config configs/cifar10_saliency_unlearn.yml \
    --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss \
    --label_to_forget 0 \
    --mode generate_mask
```
The mask is stored under `results/cifar10/unlearn/mask`.

### Step 3: Run Unlearning Optimization
Execute saliency-guided forgetting on the target class:

```bash
CUDA_VISIBLE_DEVICES="0,1" python train.py --config configs/cifar10_saliency_unlearn.yml \
    --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss \
    --label_to_forget 0 \
    --mode saliency_unlearn \
    --mask_path results/cifar10/unlearn/mask/{mask_name} \
    --alpha 1e-3 \
    --method rl
```

---

## 📊 Evaluation & Verification

### 1. Generation Quality on Retained Classes (FID / Inception Score)

Generate samples from the unlearned model excluding the forgotten class:
```bash
CUDA_VISIBLE_DEVICES="0,1" python sample.py --config configs/cifar10_sample.yml \
    --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss \
    --mode sample_fid \
    --n_samples_per_class 5000 \
    --classes_to_generate 'x0'
```

Prepare reference dataset without the forgotten class:
```bash
python save_base_dataset.py --dataset cifar10 --label_to_forget 0
```

Evaluate image metrics (FID, sFID, Precision, Recall):
```bash
CUDA_VISIBLE_DEVICES="0,1" python evaluator.py \
    results/cifar10/yyyy_mm_dd_hhmmss/fid_samples_without_label_0_guidance_2.0 \
    cifar10_without_label_0
```

### 2. Concept Erasure Verification via Classifier Audit

Fine-tune a pretrained classifier:
```bash
CUDA_VISIBLE_DEVICES="0" python train_classifier.py --dataset cifar10
```

Sample conditional outputs prompting the forgotten class:
```bash
CUDA_VISIBLE_DEVICES="0,1" python sample.py --config configs/cifar10_sample.yml \
    --ckpt_folder results/cifar10/yyyy_mm_dd_hhmmss \
    --mode sample_classes \
    --classes_to_generate "0" \
    --n_samples_per_class 500
```

Evaluate erasure efficacy using classification probability:
```bash
CUDA_VISIBLE_DEVICES="0" python classifier_evaluation.py \
    --sample_path results/cifar10/yyyy_mm_dd_hhmmss/class_samples/0 \
    --dataset cifar10 \
    --label_of_forgotten_class 0
```