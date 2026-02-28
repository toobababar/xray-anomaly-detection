# ============================================================
# inference.py
# Inference utilities for Chest X-Ray Classification.
# Used by both the CLI and the FastAPI app.
# ============================================================

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODEL_PATH, CLASS_NAMES, IMG_SIZE, NORM_MEAN, NORM_STD
from model import load_model


# ==================== PREPROCESSING ====================
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    Handles any input format (RGB, RGBA, grayscale) and converts to
    the correct tensor format expected by the model.

    Args:
        image: PIL Image object

    Returns:
        tensor of shape (1, 1, IMG_SIZE, IMG_SIZE) ready for model input
    """
    # Convert to grayscale regardless of input format
    image = image.convert("L")

    # PIL -> numpy -> tensor
    img_array = np.array(image, dtype=np.float32)
    tensor    = torch.from_numpy(img_array)

    # Ensure shape (1, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    # Normalize to [0, 1]
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    # Apply transforms and add batch dimension
    tensor = inference_transform(tensor).unsqueeze(0)

    return tensor


def predict(image: Image.Image, model) -> dict:
    """
    Run inference on a single PIL image.

    Args:
        image: PIL Image object
        model: loaded ChestXrayResNet in eval mode

    Returns:
        dict with predicted class, confidence, and all class probabilities
    """
    tensor = preprocess_image(image).to(DEVICE)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        pred  = probs.argmax().item()

    return {
        "predicted_class": CLASS_NAMES[pred],
        "confidence":      round(probs[pred].item(), 4),
        "probabilities":   {
            name: round(probs[i].item(), 4)
            for i, name in enumerate(CLASS_NAMES)
        }
    }


# ==================== CLI USAGE ====================
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image      = Image.open(image_path)
    model      = load_model(MODEL_PATH)
    result     = predict(image, model)

    print(json.dumps(result, indent=2))
