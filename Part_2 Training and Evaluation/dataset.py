import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from config import (
    LABEL_MAP, IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    AUG_ROTATION, AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_HUE, 
    BATCH_SIZE, TEST_SPLIT
)

class CRCDataset(Dataset):
    def __init__(self, folder_path, transform=None, is_training=True):
        self.folder_path = folder_path
        self.image_files = []
        self.labels = []
        self.is_training = is_training
        
        # Define transformations based on training/testing
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(AUG_ROTATION),
                    transforms.ColorJitter(
                        brightness=AUG_BRIGHTNESS,
                        contrast=AUG_CONTRAST,
                        saturation=AUG_SATURATION,
                        hue=AUG_HUE
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
                ])
        else:
            self.transform = transform
        
        # Use the label map from config
        self.label_map = LABEL_MAP

        # Walk through the directory to find all class folders
        for root, dirs, files in os.walk(folder_path):
            class_name = os.path.basename(root)
            if class_name in self.label_map:  # Only consider folders that match our classes
                label = self.label_map[class_name]
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        self.image_files.append(os.path.join(root, f))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
	
        img_path = self.image_files[idx]
	
        img = Image.open(img_path).convert("RGB")
	
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_data_loaders(folder_path, batch_size=BATCH_SIZE):
    """Create train and test data loaders with 80:20 split"""
    # Create full dataset
    full_dataset = CRCDataset(folder_path, is_training=True)
    
    # Calculate split sizes (80% train, 20% test)
    dataset_size = len(full_dataset)
    test_size = int(TEST_SPLIT * dataset_size)  # TEST_SPLIT should be 0.2 in config.py
    train_size = dataset_size - test_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override transforms for test set
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
    ])
    
    # Create a new dataset with the same data but different transforms for testing
    test_dataset_with_transforms = CRCDataset(folder_path, transform=test_transforms, is_training=False)
    test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else range(train_size, dataset_size)
    test_dataset = torch.utils.data.Subset(test_dataset_with_transforms, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    return train_loader, test_loader