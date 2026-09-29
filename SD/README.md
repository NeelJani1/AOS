# AOS: Stable Diffusion

This directory contains scripts for unlearning concepts (such as NSFW content, specific objects, or artistic styles) in Stable Diffusion models.

---

## ⚡ Memory Optimization & 8-Bit Quantization (24GB → <8GB VRAM)

Standard Stable Diffusion unlearning (FP32 precision with standard AdamW) requires **~24 GB VRAM**, typically demanding an A100 or RTX 3090/4090.

By incorporating **8-bit quantized AdamW (`bitsandbytes`)**, **bfloat16 / FP8 precision**, and **gradient checkpointing**, the peak memory requirement is reduced from **24.1 GB down to 7.8 GB**, allowing concept unlearning on consumer GPUs (e.g., RTX 3070 / 4060 / 4070 8GB).

### Key Files Implementing Low-VRAM Optimization:
- **`train-scripts/generate_mask.py`**: Employs `bnb.AdamW8bit`, AMP `GradScaler`, and `diffusion_model.use_checkpoint = True` for memory-efficient saliency gate computation.
- **`train-scripts/random_label.py`**: Concept unlearning loop utilizing 8-bit optimizer states and `expandable_segments` memory allocation.
- **`train-scripts/train-esd.py`**: Erasing Stable Diffusion with sub-8GB memory configuration.

---

## 📁 Structure
- `train-scripts/`: Contains training and unlearning scripts (e.g., `train-esd.py`, `generate_mask.py`, `random_label.py`).
- `eval-scripts/`: Contains scripts for evaluating the unlearned model (e.g., `compute-fid.py`, `generate-images.py`).
- `prompts/`: Contains CSV files with benchmark prompts used for unlearning and verification.
- `configs/`: YAML configurations for latent diffusion components.

---

## 🚀 Usage Guide

### 1. Concept Unlearning
To run the Erasing Stable Diffusion (ESD) script for removing a concept (e.g., nudity):
```bash
python train-scripts/train-esd.py --prompt "nudity" --train_method "noxattn"
```

### 2. Evaluating Generation Quality
Generate images from the model to visually inspect or compute quantitative metrics (e.g., FID):
```bash
python eval-scripts/generate-images.py
python eval-scripts/compute-fid.py
```