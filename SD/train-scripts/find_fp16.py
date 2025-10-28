import torch
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Aggressive debug patch
original_module_half = torch.nn.Module.half
original_tensor_half = torch.Tensor.half

call_count = 0

def debug_module_half(self):
    global call_count
    call_count += 1
    import traceback
    print(f"🚨 CALL #{call_count}: .half() called on module!")
    print("Stack trace (most recent call last):")
    
    # Print the relevant stack frames
    stack = traceback.extract_stack()
    for i, frame in enumerate(stack[-10:-1]):  # Last 10 frames
        filename = os.path.basename(frame.filename)
        if 'torch/' not in filename:  # Skip torch internal calls
            print(f"  {i}: {filename}:{frame.lineno} in {frame.name}")
            if frame.line:
                print(f"     {frame.line}")
    print("=" * 60)
    return self

def debug_tensor_half(self):
    global call_count
    call_count += 1
    import traceback
    print(f"🚨 CALL #{call_count}: .half() called on tensor!")
    print("Stack trace (most recent call last):")
    
    stack = traceback.extract_stack()
    for i, frame in enumerate(stack[-8:-1]):
        filename = os.path.basename(frame.filename)
        if 'torch/' not in filename:
            print(f"  {i}: {filename}:{frame.lineno} in {frame.name}")
            if frame.line:
                print(f"     {frame.line}")
    print("=" * 60)
    return self

torch.nn.Module.half = debug_module_half
torch.Tensor.half = debug_tensor_half

print("🔍 Debug patches installed - tracking all .half() calls")

# Now import and run the model setup
from dataset import setup_model

device = 'cuda:0'
config_path = '../configs/stable-diffusion/v1-inference.yaml'
ckpt_path = '../models/ldm/stable-diffusion-v1/.cache/huggingface/download/sd-v1-4.ckpt'

print(f"Loading model from: {ckpt_path}")
model = setup_model(config_path, ckpt_path, device)

print(f"✅ Model loaded successfully!")
print(f"Final model dtype: {next(model.parameters()).dtype}")
print(f"Total .half() calls intercepted: {call_count}")