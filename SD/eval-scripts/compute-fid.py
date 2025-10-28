import argparse

import torch
from dataset import setup_fid_data
from torch.utils.data import DataLoader
# Make sure this import is correct for your torchmetrics version
# If you are on 0.6.0, this is correct:
from torchmetrics.image.fid import FID
# If you are on a newer version, you might need this:
# from torchmetrics.image import FrechetInceptionDistance as FID


def compute_fid(class_to_forget, path, image_size):
    # Set a reasonable batch size
    BATCH_SIZE = 1024

    fid = FID(feature=64)
    
    # 1. Get dataset objects from our modified setup_fid_data
    real_dataset, fake_dataset = setup_fid_data(class_to_forget, path, image_size)

    # 2. Create DataLoaders to process in batches
    real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, num_workers=0)
    fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE, num_workers=0)

    print("Updating FID with real images (in batches)...")
    # 3. Process real images in batches
    for images, _ in real_loader:
        # The transform normalizes images to [-1, 1].
        # We must un-normalize back to [0, 255] for the FID metric.
        images = ((images * 0.5 + 0.5) * 255).to(torch.uint8).cpu()
        fid.update(images, real=True)

    print("Updating FID with fake images (in batches)...")
    # 4. Process fake images in batches
    for images, _ in fake_loader:
        # Also un-normalize from [-1, 1] to [0, 255]
        images = ((images * 0.5 + 0.5) * 255).to(torch.uint8).cpu()
        fid.update(images, real=False)

    print("Computing FID score...")
    # 5. Compute final score
    print(fid.compute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--folder_path", help="path of images", type=str, required=True)
    parser.add_argument(
        "--class_to_forget", help="class_to_forget", type=int, required=False, default=6
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    args = parser.parse_args()

    path = args.folder_path
    class_to_forget = args.class_to_forget
    image_size = args.image_size
    print(class_to_forget)
    compute_fid(class_to_forget, path, image_size)