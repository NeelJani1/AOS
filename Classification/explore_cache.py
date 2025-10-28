import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def extract_hf_dataset_to_dali():
    cache_dir = "/mnt/d/HF_Cache/huggingface"
    output_path = "/mnt/d/ImageNet_DALI"
    
    # Create output directories
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Extracting from HF cache: {cache_dir}")
    print(f"Output to: {output_path}")
    
    try:
        # Load train dataset
        print("\nLoading training dataset...")
        train_dataset = load_dataset(
            "imagenet-1k",
            split="train",
            cache_dir=cache_dir,
            token=True,
            streaming=False  # Use False to load full dataset
        )
        
        print(f"Training samples: {len(train_dataset)}")
        
        # Extract training images
        successful_train = 0
        for idx, sample in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="Extracting train"):
            try:
                image = sample['image']
                label = sample['label']
                
                # Create class folder
                class_folder = f"n{label:08d}"
                class_dir = os.path.join(train_dir, class_folder)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save image
                filename = f"img_{idx:08d}_{label}.JPEG"
                filepath = os.path.join(class_dir, filename)
                
                if isinstance(image, Image.Image):
                    image.save(filepath, 'JPEG', quality=95)
                    successful_train += 1
                else:
                    # Convert if not PIL image
                    pil_image = Image.fromarray(image)
                    pil_image.save(filepath, 'JPEG', quality=95)
                    successful_train += 1
                    
            except Exception as e:
                if successful_train < 5:  # Show first few errors
                    print(f"Train error at {idx}: {e}")
                continue
        
        print(f"Successfully extracted {successful_train}/{len(train_dataset)} training images")
        
    except Exception as e:
        print(f"Error loading training dataset: {e}")
    
    try:
        # Load validation dataset
        print("\nLoading validation dataset...")
        val_dataset = load_dataset(
            "imagenet-1k",
            split="validation",
            cache_dir=cache_dir,
            token=True,
            streaming=False
        )
        
        print(f"Validation samples: {len(val_dataset)}")
        
        # Extract validation images
        successful_val = 0
        for idx, sample in tqdm(enumerate(val_dataset), total=len(val_dataset), desc="Extracting val"):
            try:
                image = sample['image']
                label = sample['label']
                
                # Create class folder
                class_folder = f"n{label:08d}"
                class_dir = os.path.join(val_dir, class_folder)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save image
                filename = f"img_{idx:08d}_{label}.JPEG"
                filepath = os.path.join(class_dir, filename)
                
                if isinstance(image, Image.Image):
                    image.save(filepath, 'JPEG', quality=95)
                    successful_val += 1
                else:
                    # Convert if not PIL image
                    pil_image = Image.fromarray(image)
                    pil_image.save(filepath, 'JPEG', quality=95)
                    successful_val += 1
                    
            except Exception as e:
                if successful_val < 5:  # Show first few errors
                    print(f"Val error at {idx}: {e}")
                continue
        
        print(f"Successfully extracted {successful_val}/{len(val_dataset)} validation images")
        
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Train images: {count_images(train_dir)}")
    print(f"Val images: {count_images(val_dir)}")

def count_images(directory):
    """Count number of images in directory structure"""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                count += 1
    return count

if __name__ == "__main__":
    extract_hf_dataset_to_dali()