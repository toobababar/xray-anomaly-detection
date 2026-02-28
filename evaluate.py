# ============================================================
# evaluate.py
# Evaluation script for Chest X-Ray Classification.
# Run: python evaluate.py
# ============================================================

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import DEVICE, MODEL_PATH, CLASS_NAMES
from dataset import build_dataloaders
from model import load_model


def evaluate(model, test_loader):
    """
    Run model on test set and return predictions and true labels.

    Args:
        model:       trained model in eval mode
        test_loader: DataLoader for test set

    Returns:
        all_preds, all_labels as plain Python lists
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
            outputs = model(images.to(DEVICE))
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    print(f"Evaluating on: {DEVICE}")

    # Load best model
    model = load_model(MODEL_PATH)

    # Build dataloaders — we only need test_loader here
    _, _, test_loader, _ = build_dataloaders()

    # Run evaluation
    all_preds, all_labels = evaluate(model, test_loader)

    # Print classification report
    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES)
