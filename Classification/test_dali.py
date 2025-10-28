import os
from imagenet import find_imagenet_folders, get_imagenet_iter_dali

def test_dali_pipeline():
    print("Testing DALI pipeline...")
    
    # Check if we can find ImageNet folders
    imagenet_path = find_imagenet_folders()
    if imagenet_path:
        print(f"✓ Found ImageNet at: {imagenet_path}")
        
        # Test if DALI can load data
        try:
            train_loader = get_imagenet_iter_dali(
                'train', 
                imagenet_path, 
                batch_size=32, 
                num_threads=4,
                device_id=0
            )
            
            # Try to get one batch
            data = next(iter(train_loader))
            print(f"✓ DALI pipeline working!")
            print(f"  Batch shape: {data[0]['data'].shape}")
            print(f"  Labels shape: {data[0]['label'].shape}")
            
        except Exception as e:
            print(f"✗ DALI error: {e}")
            print("This means DALI cannot read the folder structure.")
            
    else:
        print("✗ No ImageNet folder structure found.")
        print("DALI requires this specific folder structure:")
        print("  /path/to/imagenet/train/n01440764/xxx.JPEG")
        print("  /path/to/imagenet/val/n01440764/xxx.JPEG")

if __name__ == "__main__":
    test_dali_pipeline()