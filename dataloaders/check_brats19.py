import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from brats19 import * # Ensure this points to your updated BraTS class
# from utils import RandomRotFlip, RandomCrop, ToTensor  # Ensure these are implemented

def check_dataset(dataset_path):
    """
    Check the BraTS dataset by loading a sample and printing its shape.
    """
    # Set up transforms
    transform = transforms.Compose([
        RandomRotFlip(),
        RandomCrop(output_size=(96, 96, 96)),
        ToTensor(),
    ])
    
    # Initialize dataset
    print("Initializing dataset...")
    db_train = BraTS(base_dir=dataset_path, split='train', transform=transform)
    
    # Test dataset size
    print(f"Dataset contains {len(db_train)} samples.")

    # Get a sample from the dataset
    sample = db_train[0]
    print(f"Sample loaded. Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}")

    # Check the data loading mechanism by using a DataLoader
    batch_size = 8  # Adjust as needed
    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)

    print(f"Batch size: {batch_size}")
    print("Loading first batch...")

    for i, batch in enumerate(train_loader):
        if i == 0:  # Just check the first batch
            print(f"Batch {i} loaded. Image batch shape: {batch['image'].shape}, Label batch shape: {batch['label'].shape}")
            break

    print("Dataset and batch checking complete.")

if __name__ == "__main__":
    dataset_path = r"D:\Dataset\BraTS_2019\BraTS_Flair_seg\BraTS_2019"
    
    if not os.path.exists(dataset_path):
        print(f"Error: The dataset path {dataset_path} does not exist.")
        sys.exit(1)
    
    check_dataset(dataset_path)
