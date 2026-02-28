# ============================================================
# train.py
# Training script for Chest X-Ray Classification.
# Run: python train.py
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    DEVICE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, MODEL_PATH
)
from dataset import build_dataloaders
from model import ChestXrayResNet


def train_model(model, train_loader, val_loader):
    """
    Full training loop with validation, LR scheduling, and checkpointing.

    Args:
        model:        ChestXrayResNet instance
        train_loader: DataLoader for training set
        val_loader:   DataLoader for validation set

    Returns:
        history dict with train/val loss and accuracy per epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Halve LR when val loss stops improving for LR_PATIENCE epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE,
        factor=LR_FACTOR, verbose=True
    )

    model.to(DEVICE)
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):

        # -------- TRAINING PHASE --------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # -------- VALIDATION PHASE --------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Save only when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")

    return history


def plot_history(history):
    """Plot training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved to training_curves.png")


if __name__ == "__main__":
    print(f"Training on: {DEVICE}")

    # Build dataloaders
    train_loader, val_loader, _, _ = build_dataloaders()

    # Initialize model
    model = ChestXrayResNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready. Trainable parameters: {total_params:,}")

    # Train
    history = train_model(model, train_loader, val_loader)

    # Plot curves
    plot_history(history)

    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
