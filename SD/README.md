# AOS: Stable Diffusion

This directory contains scripts for unlearning concepts (such as NSFW content, specific objects, or artistic styles) in Stable Diffusion models.

## Structure
- `train-scripts/`: Contains training and unlearning scripts (e.g., `train-esd.py`, `gradient_ascent.py`, `nsfw_removal.py`).
- `eval-scripts/`: Contains scripts for evaluating the unlearned model (e.g., `compute-fid.py`, `generate-images.py`).
- `prompts/`: Contains CSV files with unsafe and standard prompts used for unlearning and evaluation.
- `configs/`: YAML configurations for the latent diffusion and autoencoder components.

## Usage Guide

### Concept Unlearning
To run the Erasing Stable Diffusion (ESD) script for removing a concept (e.g., nudity):
```bash
python train-scripts/train-esd.py --prompt "nudity" --train_method "noxattn"
```

### Evaluating Generation Quality
Generate images from the model to visually inspect or compute quantitative metrics (e.g., FID):
```bash
python eval-scripts/generate-images.py
python eval-scripts/compute-fid.py
```