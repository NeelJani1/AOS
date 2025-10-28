import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(dataset_name, batch_size=128, num_workers=4):
    if dataset_name.lower() == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
        val_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_val)

    elif dataset_name.lower() == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        try:
            train_dataset = datasets.ImageFolder(root='./data/imagenet/train', transform=transform_train)
            val_dataset = datasets.ImageFolder(root='./data/imagenet/val', transform=transform_val)
        except Exception as e:
            from datasets import load_dataset
            print("⚠️ Local ImageNet not found. Using Hugging Face streaming version...")
            dataset = load_dataset("imagenet-1k", split="train", streaming=True)
            val_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
            return dataset, val_dataset, True  # streaming flag

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, False
