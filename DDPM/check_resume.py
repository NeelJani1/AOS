import torch
import os

# Path to your checkpoint (adjust based on step 2)
ckpt_path = "./results/cifar10/2025_10_19_101846/ckpts/ckpt.pth"

if os.path.exists(ckpt_path):
    print(f"✅ Checkpoint found: {ckpt_path}")
    states = torch.load(ckpt_path, map_location='cpu')
    print(f"✅ Checkpoint loaded successfully")
    print(f"✅ Number of states: {len(states)}")
    if len(states) >= 3:
        print(f"✅ Should resume from step: {states[2]}")
    else:
        print("❌ Checkpoint doesn't have step information")
else:
    print(f"❌ Checkpoint not found at: {ckpt_path}")