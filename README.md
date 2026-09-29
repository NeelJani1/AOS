# Adaptive Otsu Unlearning (AOS): A Variance-Aware Framework for Stable and Interpretable Machine Unlearning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Institution](https://img.shields.io/badge/Institution-University%20of%20Auckland-003366.svg)](https://www.auckland.ac.nz/)

> **Authors**: [Gurudas Salunke](mailto:gsal919@aucklanduni.ac.nz) and [Neel Jani](mailto:njan320@aucklanduni.ac.nz)  
> **Affiliation**: Department of Computer Science, University of Auckland, New Zealand  
> **Paper**: *Adaptive Otsu Unlearning: A Variance-Aware Framework for Stable and Interpretable Machine Unlearning*

---

## Abstract

Machine unlearning (MU) has become a foundational requirement for privacy-compliant artificial intelligence systems that must "forget" data upon request. While saliency-guided unlearning methods such as **SalUn** demonstrate competitive forgetting by identifying and perturbing influential weights via gradient ascent, they rely on manually tuned, static saliency thresholds ($\tau$). This static threshold is brittle across varying architectures, datasets, and unlearning regimes.

This repository hosts the official implementation of **Adaptive Otsu Saliency (AOS)**, an automated, variance-maximizing unlearning framework that dynamically computes layer-specific thresholds derived from gradient saliency distributions using Otsu's method. AOS integrates three key mechanisms:
1. **Layer-wise Otsu Thresholding**: Statistically isolates salient from redundant weights by maximizing between-class variance.
2. **Fisher-Weighted Gradient Normalization**: Stabilizes updates along high-curvature parameter directions using the Fisher Information Matrix (FIM).
3. **Retention-Aware Dynamic Scaling & Threshold Annealing**: Balances knowledge retention and forgetting dynamically, preventing catastrophic forgetting and concept revival.

Evaluated on CIFAR-100 (ResNet-18) and generative diffusion tasks (DDPM & Stable Diffusion), AOS achieves superior or comparable unlearning accuracy (UA) with up to **10–15% higher retain accuracy (RA)**, **37% lower stability variance**, and **8% lower runtime** compared to fixed-threshold baselines.

---

## Core Contributions & Methodology

### 1. Gradient Saliency Formulation
For a model parameterized by weights $W = \{w_i\}$ and forget-set loss $\mathcal{L}_f$, raw gradient saliency is computed as:
$$s_i = \left\| \frac{\partial \mathcal{L}_f}{\partial w_i} \right\|^2$$

### 2. Fisher-Weighted Gradient Normalization
To prevent over-updating in high-curvature directions, saliencies are normalized inversely to their empirical Fisher Information $F_i$:
$$\tilde{s}_i = \frac{s_i}{\sqrt{F_i + \epsilon}}, \quad F_i \approx \mathbb{E}_{(x,y)\sim \mathcal{D}_r}\left[\left(\frac{\partial \log p(y|x; W)}{\partial w_i}\right)^2\right]$$

### 3. Layer-Wise Otsu Thresholding
Parameter saliencies $\tilde{s}^{(l)}$ in each layer $l$ are treated as a 1D distribution partitioned into redundant ($\mathcal{C}_0$) and salient ($\mathcal{C}_1$) subsets. The optimal threshold $\tau^{*(l)}$ maximizes the between-class variance $\sigma_B^2(\tau)$:
$$\tau^{*(l)} = \arg\max_{\tau} \omega_0^{(l)}(\tau)\,\omega_1^{(l)}(\tau)\left[\mu_0^{(l)}(\tau) - \mu_1^{(l)}(\tau)\right]^2$$
where $\omega_0, \omega_1$ and $\mu_0, \mu_1$ are the respective class cumulative probabilities and means. The layer-wise binary mask is:
$$M_i^{(l)} = \mathbb{I}\left[\tilde{s}_i^{(l)} > \tau_t^{*(l)}\right]$$

### 4. Retention-Aware Gradient Scaling
To preserve representations shared between forget and retain classes, update magnitudes are modulated per class $c$ using gradient energy ratios:
$$\lambda_c = \frac{\mathbb{E}_{x \in \mathcal{D}_r^c}\left[\|\nabla_W \mathcal{L}_r(x)\|\right]}{\mathbb{E}_{x \in \mathcal{D}_f}\left[\|\nabla_W \mathcal{L}_f(x)\|\right]}$$

### 5. Dynamic Threshold Annealing & Parameter Update
To handle early optimization phases before bimodal separation stabilizes, threshold annealing transitions smoothly from an initial percentile threshold $\tau_{\text{init}}$ with decay constant $T_a$:
$$\tau_t^{*(l)} = \alpha_t \tau_{\text{init}}^{(l)} + (1 - \alpha_t)\tau_{\text{Otsu}}^{(l)}, \quad \alpha_t = \exp\left(-\frac{t}{T_a}\right)$$

The final parameter update for unlearning is:
$$w_i^{(t+1)} = w_i^{(t)} + \eta \cdot \lambda_c \cdot M_i^{(l)} \cdot \frac{\partial \mathcal{L}_f}{\partial w_i^{(l)}}$$

---

## Algorithm 1: AOS Unlearning Framework

```text
Algorithm 1: Adaptive Otsu Saliency (AOS) Unlearning Framework
Require: Model W, forget dataset D_f, retain dataset D_r, learning rate eta, annealing constant T_a
1:  Initialize tau_init^(l) for all layers l
2:  for each unlearning epoch t = 1 to T do
3:      Compute saliencies s_i^(l) = || \partial L_f / \partial w_i^(l) ||^2
4:      Estimate Fisher F_i^(l) on D_r
5:      Normalize \tilde{s}_i^(l) = s_i^(l) / \sqrt{F_i^(l) + \epsilon}
6:      Construct histogram H^(l)(\tilde{s}), compute tau_Otsu^(l)
7:      Compute tau_t^{*(l)} = \alpha_t \tau_{init}^{(l)} + (1 - \alpha_t) \tau_{Otsu}^{(l)}
8:      Form mask M_i^(l) = 1[\tilde{s}_i^(l) > \tau_t^{*(l)}]
9:      Compute \lambda_c from retain-forget gradient ratio
10:     Update weights: w_i^{(t+1)} = w_i^{(t)} + \eta \lambda_c M_i^{(l)} (\partial L_f / \partial w_i^(l))
11:     Fine-tune on retain set D_r for stability
12: end for
13: return Unlearned weights W'
```

---

## Experimental Results & Benchmark Tables

### Table I: Comparison with Existing Paradigms
| Method | Type | Adaptivity | Explainability |
| :--- | :--- | :--- | :--- |
| **SISA** [1] | Certified | Static | High |
| **Eternal Sunshine** [5] | Gradient | Partial | Moderate |
| **SalUn** [9] | Gradient + Masking | Fixed | Moderate |
| **AMU** [14] | Adaptive Rate | Dynamic | Low |
| **AOS (Ours)** | **Statistical + Gradient** | **Fully Adaptive** | **High** |

---

### Table II: Retrain Baseline with Early Stopping (CIFAR-100, ResNet-18)
| Forget Split | Best TA (%) | Retain Acc (RA %) | Epochs | Training Efficiency (%) |
| :---: | :---: | :---: | :---: | :---: |
| **10%** | 65.28 | 93.72 | 41/45 | 91.1% |
| **20%** | 63.61 | 87.89 | 42/45 | 93.3% |
| **30%** | 61.19 | 89.83 | 41/45 | 91.1% |
| **40%** | 59.16 | 88.09 | 42/45 | 93.3% |
| **50%** | 55.09 | 85.61 | 41/45 | 91.1% |
| **60%** | 49.75 | 81.20 | 44/45 | 97.8% |
| **70%** | 44.82 | 74.78 | 40/45 | 88.9% |
| **80%** | 35.78 | 66.22 | 42/45 | 93.3% |
| **90%** | 24.35 | 47.36 | 38/43 | 84.4% |

---

### Table III: Comprehensive Method Comparison on CIFAR-100 (ResNet-18)
| Forget % | FT Test | FT Forget | FT Retain | GA Test | GA Forget | GA Retain | RL (Retrain) Test | RL Forget | RL Retain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 55.29 | 57.26 | 61.57 | 62.31 | 81.88 | 83.08 | 54.06 | 55.80 | 60.21 |
| **20%** | 55.09 | 56.92 | 63.96 | 21.26 | 25.01 | 25.43 | 53.75 | 53.92 | 61.62 |
| **30%** | 53.00 | 55.45 | 63.99 | 1.03 | 0.93 | 1.05 | 54.58 | 55.16 | 64.34 |
| **40%** | 54.20 | 54.72 | 64.22 | 1.07 | 0.89 | 0.97 | 53.76 | 53.49 | 64.54 |
| **50%** | 53.64 | 55.04 | 65.68 | 1.00 | 0.96 | 1.06 | 53.60 | 54.04 | 65.86 |
| **60%** | 52.78 | 54.71 | 68.48 | 1.04 | 0.97 | 1.07 | 50.15 | 50.85 | 62.98 |
| **70%** | 48.76 | 50.67 | 65.04 | 1.04 | 1.03 | 1.01 | 48.92 | 49.24 | 64.33 |
| **80%** | 46.43 | 48.34 | 70.33 | 1.03 | 1.03 | 0.93 | 47.62 | 48.65 | 67.18 |
| **90%** | 43.05 | 44.59 | 67.14 | 1.00 | 0.99 | 1.10 | 39.53 | 39.56 | 60.00 |

---

### Table IV: Ablation Study at 50% Forget Ratio
| Variant | Description | Forget Acc (%) | Retain Acc (%) | RA Stability Variance ($\sigma^2$) |
| :--- | :--- | :---: | :---: | :---: |
| **SalUn Baseline** | Fixed static threshold $\tau$ | 8.3 | 77.2 | 0.024 |
| **AOS-T** | Otsu Thresholding Only | 8.0 | 80.6 | 0.018 |
| **AOS-F** | Fisher Normalization Only | 7.9 | 81.8 | 0.016 |
| **AOS-R** | Retention Scaling Only | 8.2 | 82.5 | 0.015 |
| **AOS (Full)** | Complete Framework | **7.8** | **83.4** | **0.014** |

> **Key Takeaway**: At 50% forgetting, AOS achieves **7.8% Forget Accuracy** while sustaining **83.4% Retain Accuracy** (+6.2% over SalUn and +4.3% over AMU), with a **37% reduction in variance** and **8% runtime reduction** via earlier convergence.

---

## Directory Layout

```
AOS/
├── Classification/              # Classification unlearning (ResNet-18, VGG)
│   ├── models/                  # Network architectures
│   ├── otsu_utils.py            # Core AOS: Bounded Otsu, Fisher norm, retention bounds
│   ├── main_train.py            # Baseline training script
│   ├── main_unlearning.py       # AOS unlearning loop
│   ├── run_otsu_experiments.py  # Automation for multi-ratio evaluations
│   ├── mia_evaluation.py        # Membership Inference Attack (MIA) evaluation
│   ├── requirements.txt         # Classification-specific requirements
│   └── README.md                # Detailed guide for image classification
│
├── DDPM/                        # Generative unlearning: Classifier-Free Guidance DDPM
│   ├── configs/                 # YAML configurations (train, sample, forget, FIM)
│   ├── models/                  # Diffusion model definitions (UNet, EMA)
│   ├── functions/               # Denoising, loss, and checkpoint utilities
│   ├── train.py                 # DDPM training and unlearning entry point
│   ├── quick_eval.py            # Sample generation and FID evaluation
│   ├── classifier_evaluation.py # Unlearning verification on generated samples
│   └── README.md                # Guide for generative DDPM unlearning
│
├── SD/                          # Stable Diffusion concept erasure
│   ├── train-scripts/           # Erasing Stable Diffusion (ESD) scripts
│   ├── eval-scripts/            # FID, CLIP, and visual generation evaluations
│   ├── configs/                 # Latent diffusion configuration files
│   ├── prompts/                 # Benchmark evaluation and unlearning prompts
│   └── README.md                # Guide for Stable Diffusion unlearning
│
├── images/                      # Figures, architectural schematics, and plots
├── requirements.txt             # Unified Python dependencies
├── environment.yml              # Conda environment definition
└── README.md                    # Root project documentation
```

---

## Installation & Setup

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/NeelJani1/AOS.git
cd AOS

# Create and activate conda environment
conda env create -f environment.yml
conda activate aos
```

### Option 2: Pip Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Quickstart & Reproduction Guide

### 1. Image Classification (CIFAR-100 / ResNet-18)

#### Step 1: Train Baseline Model
```bash
cd Classification
python main_train.py --arch resnet18 --dataset cifar100 --epochs 100 --lr 0.1 --save_dir ./weights
```

#### Step 2: Run AOS Unlearning
```bash
# Run AOS unlearning across 10% forget ratio
python run_otsu_experiments.py --method FT --forget_ratio 0.1

# Run with specific unlearning objective (GA, FT, or RL)
python main_unlearning.py --arch resnet18 --dataset cifar100 --method FT --forget_ratio 0.5 --anneal_epochs 5
```

#### Step 3: Evaluate Unlearned Model (Accuracy & MIA)
```bash
python evaluate_model.py --model_path ./weights/unlearned_model.pt --dataset cifar100
python mia_evaluation.py --model_path ./weights/unlearned_model.pt --dataset cifar100
```

---

### 2. Generative Diffusion Unlearning (DDPM)

```bash
cd DDPM

# Train baseline DDPM on CIFAR-10
python train.py --config configs/cifar10_train.yml

# Execute saliency-guided forgetting on selected classes
python train.py --config configs/cifar10_saliency_unlearn.yml

# Evaluate image generation quality & sample post-unlearning
python quick_eval.py --config configs/cifar10_sample.yml
python classifier_evaluation.py
```

---

### 3. Stable Diffusion Concept Erasure

```bash
cd SD

# Erase unwanted visual concepts (e.g., nudity or specific artists)
python train-scripts/train-esd.py --prompt "nudity" --train_method "noxattn" --devices "0,0"

# Generate evaluation images to test concept erasure
python eval-scripts/generate-images.py --prompts prompts/test_prompts.csv
python eval-scripts/compute-fid.py
```

---

## Citation

If you find this work or codebase helpful in your research, please cite our paper:

```bibtex
@article{salunke2024adaptive,
  title={Adaptive Otsu Unlearning: A Variance-Aware Framework for Stable and Interpretable Machine Unlearning},
  author={Salunke, Gurudas and Jani, Neel},
  journal={Department of Computer Science, University of Auckland},
  year={2024}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
