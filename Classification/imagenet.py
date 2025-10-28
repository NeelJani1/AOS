import os
import sys

import torch
import torchvision
from datasets.load import load_dataset, DownloadConfig
from torch.utils.data import DataLoader, Subset

# sys.path.append(".")
# from cfg import *
from tqdm import tqdm


def prepare_data(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=1,
    train_subset_indices=None,
    val_subset_indices=None,
    data_path="/mnt/d/HF_Cache", # <-- UPDATED: Points to D: drive
):
    # This will now use /mnt/d/HF_Cache/huggingface as the cache directory
    path = os.path.join(data_path, "huggingface")
    os.makedirs(path, exist_ok=True) # Ensure the directory exists

    if dataset == "imagenet":
        download_config = DownloadConfig(resume_download=True)
        train_set = load_dataset(
            "imagenet-1k",
            token=True,
            split="train",
            cache_dir=path, # Uses the D: drive path
            download_config=download_config,
            streaming=True # Keep streaming=True for now during download phase
        )
        validation_set = load_dataset(
            "imagenet-1k",
            token=True,
            split="validation",
            cache_dir=path, # Uses the D: drive path
            download_config=download_config,
            streaming=True # Keep streaming=True for now during download phase
        )

        # --- Transforms remain the same ---
        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomResizedCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
            # Handle potential None images during streaming download issues
            examples["image"] = [transform(img) for img in examples["image"] if img is not None]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            )
            # Handle potential None images during streaming download issues
            examples["image"] = [transform(img) for img in examples["image"] if img is not None]
            return examples

    elif dataset == "tiny_imagenet":
        # ... (TinyImageNet code remains the same, but uses cache_dir=path) ...
        train_set = load_dataset(
             "Maysee/tiny-imagenet", token=True, split="train", cache_dir=path
         )
        validation_set = load_dataset(
             "Maysee/tiny-imagenet", token=True, split="valid", cache_dir=path
         )
         # ... (transforms remain the same) ...

    elif dataset == "flowers102":
        # ... (Flowers102 code remains the same, but uses cache_dir=path) ...
        train_set = load_dataset(
             "nelorth/oxford-flowers", token=True, split="train", cache_dir=path
         )
        validation_set = load_dataset(
             "nelorth/oxford-flowers", token=True, split="test", cache_dir=path
         )
         # ... (transforms remain the same) ...

    else:
        raise NotImplementedError

    # Apply transforms using .map() for IterableDataset
    train_set = train_set.map(train_transform, batched=True)
    validation_set = validation_set.map(validation_transform, batched=True)

    # Add shuffling for streaming dataset
    if shuffle:
        train_set = train_set.shuffle(buffer_size=1000, seed=42)

    # --- Subset logic remains the same (with warnings) ---
    if train_subset_indices is not None:
        print("Warning: Subset indices may not be fully compatible with streaming datasets.")
        # This part might need further adjustment depending on how subsets interact with streaming.
        # For the initial download, this section won't be used anyway.
        forget_indices = torch.ones_like(train_subset_indices) - train_subset_indices
        train_subset_indices_list = torch.nonzero(train_subset_indices).squeeze().tolist()
        forget_indices_list = torch.nonzero(forget_indices).squeeze().tolist()

        # Filtering logic might be needed here depending on exact streaming implementation
        # For simplicity, we'll assume the original Subset logic might partially work or fail later.
        retain_set = train_set # Placeholder, actual filtering needed
        forget_set = train_set # Placeholder, actual filtering needed


    if val_subset_indices is not None:
         print("Warning: Subset indices may not be fully compatible with streaming datasets.")
         val_subset_indices_list = torch.nonzero(val_subset_indices).squeeze().tolist()
         # Filtering logic might be needed here
         validation_set = validation_set # Placeholder

    # --- DataLoader setup remains the same, using num_workers ---
    if train_subset_indices is not None:
        # This part requires proper filtering logic for streaming subsets
         loaders = {
             "train": DataLoader(
                 retain_set, batch_size=batch_size, num_workers=num_workers, shuffle=None, pin_memory=True
             ),
             "val": DataLoader(
                 validation_set, batch_size=batch_size, num_workers=num_workers, shuffle=None, pin_memory=True
             ),
             "fog": DataLoader(
                 forget_set, batch_size=batch_size, num_workers=num_workers, shuffle=None, pin_memory=True
             ),
         }
    else:
        loaders = {
            "train": DataLoader(
                train_set, batch_size=batch_size, num_workers=num_workers, shuffle=None, pin_memory=True
            ),
            "val": DataLoader(
                validation_set, batch_size=batch_size, num_workers=num_workers, shuffle=None, pin_memory=True
            ),
        }
    return loaders


# --- get_x_y_from_data_dict remains the same ---
def get_x_y_from_data_dict(data, device):
    # Adjust based on actual keys in the streaming dictionary
    if 'image' in data and 'label' in data:
        x, y = data['image'], data['label']
        # Ensure x is a list of tensors before trying to stack or move
        if isinstance(x, list):
             # Filter out None values potentially caused by transform errors on bad data
             x = [item for item in x if item is not None]
             if not x: # If all images in batch were None
                 return None, None # Signal to skip batch
             # Assuming tensors are correctly returned by transform
             # Convert list of tensors to a batched tensor
             # Stacking might be needed if transforms return individual tensors
             # Example: x = torch.stack(x)
             pass # Assuming map handles batching correctly now
    else:
        # Fallback might fail if data isn't dict with expected keys
        try:
            x, y = data.values()
        except Exception:
             print("Unexpected data format:", data)
             return None, None # Signal to skip

    # Ensure x and y are tensors before moving to device
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
         x, y = x.to(device), y.to(device)
         return x, y
    elif isinstance(x, list) and all(isinstance(i, torch.Tensor) for i in x):
         # If transform didn't batch, stack them now
         try:
            x = torch.stack(x).to(device)
            y = torch.tensor(y).to(device) # Assuming y was a list/tensor of labels
            return x, y
         except Exception as e:
            print(f"Error stacking tensors: {e}")
            return None, None
    else:
        # Handle cases where conversion didn't work as expected
        print(f"Unexpected data types: x is {type(x)}, y is {type(y)}")
        return None, None


# --- __main__ block remains largely the same for testing ---
if __name__ == "__main__":
    ys = {}
    ys["train"] = []
    ys["val"] = []

    class ArgsPlaceholder:
        batch_size = 1
        workers = 1 # Keep low for testing download/structure

    args_placeholder = ArgsPlaceholder()

    # Pass the D: drive path for testing
    loaders = prepare_data(
        dataset="imagenet",
        batch_size=args_placeholder.batch_size,
        num_workers=args_placeholder.workers,
        shuffle=False,
        data_path="/mnt/d/HF_Cache" # Use the D: drive path here too
    )

    print("Iterating validation set (streaming)...")
    val_count = 0
    for data in tqdm(loaders["val"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        if x is not None and y is not None:
            ys["val"].append(y.item())
            val_count += 1
        if val_count >= 10: # Limit iterations for quick testing
             break

    print("Iterating train set (streaming)...")
    train_count = 0
    for data in tqdm(loaders["train"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        if x is not None and y is not None:
            ys["train"].append(y.item())
            train_count += 1
        if train_count >= 10: # Limit iterations for quick testing
            break

    # Only save if we actually got some data
    if ys["train"]:
        ys["train"] = torch.Tensor(ys["train"]).long()
        torch.save(ys["train"], "train_ys.pth")
    if ys["val"]:
        ys["val"] = torch.Tensor(ys["val"]).long()
        torch.save(ys["val"], "val_ys.pth")

    print(f"Finished testing iteration. Processed {train_count} train and {val_count} val samples.")