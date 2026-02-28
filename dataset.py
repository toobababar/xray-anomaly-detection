# ============================================================
# dataset.py
# Dataset class and DataLoader factory for Chest X-Ray Classification.
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import (
    DATA_PATH, IMG_SIZE, NORM_MEAN, NORM_STD,
    VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED
)


# ==================== TRANSFORMS ====================
# Augmentation for training only — helps prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# No augmentation for validation, test, and inference
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


# ==================== DATASET ====================
class ChestXrayNPZDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for large .npz files.
    """
    def __init__(self, npz_path=DATA_PATH, transform=None):
        self.transform = transform

        # Open once and keep handle alive for dataset lifetime
        self.data   = np.load(npz_path, mmap_mode='r')

        # Store direct array references — resolves dictionary key only once
        self.images = self.data['image']
        self.labels = self.data['image_label']
        self.length = self.images.shape[0]

        print(f"Dataset ready: {self.length} images")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Direct array index — no key lookup overhead
        image = torch.tensor(np.array(self.images[idx]), dtype=torch.float32)

        # .item() extracts plain Python scalar — avoids NumPy 1.25 DeprecationWarning
        label = torch.tensor(self.labels[idx].item(), dtype=torch.long)

        # (H, W, C) -> (C, H, W) — PyTorch expects channel first
        image = image.permute(2, 0, 1)

        # Average RGB channels to grayscale (C, H, W) -> (1, H, W)
        image = image.mean(dim=0, keepdim=True)

        # Normalize pixel values from [0, 255] to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label

    def __del__(self):
        # Close file handle cleanly when dataset object is destroyed
        if hasattr(self, 'data'):
            self.data.close()


# ==================== SUBSET WITH TRANSFORM OVERRIDE ====================
class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a different transform.
    Used to apply val_transform to val/test splits without
    affecting the original dataset's transform.
    """
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label


# ==================== DATALOADER FACTORY ====================
def build_dataloaders(npz_path=DATA_PATH):
    """
    Builds train, validation, and test DataLoaders from the .npz file.

    Returns:
        train_loader, val_loader, test_loader, full_dataset
    """
    full_dataset = ChestXrayNPZDataset(npz_path, transform=train_transform)

    n       = len(full_dataset)
    n_val   = int(n * VAL_SPLIT)
    n_test  = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Override transforms for val and test — no augmentation
    val_ds  = TransformSubset(val_ds,  val_transform)
    test_ds = TransformSubset(test_ds, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"Split -> Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader, full_dataset
