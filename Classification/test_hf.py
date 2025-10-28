import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def test_small_extraction():
    cache_dir = "/mnt/d/HF_Cache/huggingface"
    output_path = "/mnt/d/ImageNet_DALI_Test"
    
    # Create output directories
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("Testing extraction with first 100 samples...")
    
    try:
        # Load small subset of train dataset
        train_dataset = load_dataset(
            "imagenet-1k",
            split="train",
            cache_dir=cache_dir,
            token=True,
            streaming=True  # Use streaming for test
        ).take(100)
        
        successful = 0
        for idx, sample in enumerate(train_dataset):
            try:
                image = sample['image']
                label = sample['label']
                
                class_folder = f"n{label:08d}"
                class_dir = os.path.join(train_dir, class_folder)
                os.makedirs(class_dir, exist_ok=True)
                
                filename = f"test_img_{idx:08d}_{label}.JPEG"
                filepath = os.path.join(class_dir, filename)
                
                if isinstance(image, Image.Image):
                    image.save(filepath, 'JPEG')
                    successful += 1
                else:
                    pil_image = Image.fromarray(image)
                    pil_image.save(filepath, 'JPEG')
                    successful += 1
                    
            except Exception as e:
                print(f"Error at {idx}: {e}")
                continue
        
        print(f"Successfully extracted {successful}/100 test images")
        print(f"Test output: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_small_extraction()