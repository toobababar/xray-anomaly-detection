# ============================================================
# model.py
# ResNet50 model architecture for Chest X-Ray Classification.
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE


class ChestXrayResNet(nn.Module):
    """
    ResNet50 adapted for grayscale chest X-ray classification.

    Key adaptations from standard ResNet50:
    1. First conv layer changed from 3-channel to 1-channel input
       for grayscale images. Pretrained RGB weights are averaged
       across channels to preserve learned features.
    2. Final FC layer replaced with a custom classifier head
       with dropout for regularization.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Load pretrained ResNet50 — modern weights API
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Adapt first conv for grayscale (1 channel instead of 3)
        # Average pretrained RGB weights to preserve learned features
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        # Replace final classifier for num_classes output
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def load_model(model_path, device=DEVICE):
    """
    Load a trained model from a saved state dict.

    Args:
        model_path: path to saved .pth file
        device:     device to load model onto

    Returns:
        model in eval mode on the specified device
    """
    model = ChestXrayResNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model
