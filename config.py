# ============================================================
# config.py
# Central configuration for the Chest X-Ray Classification project.
#
# PRODUCTION (Docker): only MODEL_DIR and MODEL_PATH are used
# TRAINING (Colab/local):    DATA_PATH is additionally required           
#                            Run scripts/download_data.py first
# ============================================================
import torch
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data" / "5194114.npz"
MODEL_DIR   = BASE_DIR / "models"
MODEL_PATH  = MODEL_DIR / "resnet50_chestxray_v1.pth"
ONNX_PATH   = MODEL_DIR / "resnet50_chestxray_v1.onnx"

# Create model directory if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DATASET ====================
CLASS_NAMES = [
    "covid",
    "lung_opacity",
    "normal",
    "viral_pneumonia",
    "tuberculosis"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224

# Dataset split ratios
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.10
SEED       = 42   # fixed seed for reproducible splits

# ==================== TRAINING ====================
BATCH_SIZE   = 16
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
LR_PATIENCE = 3
LR_FACTOR   = 0.5

# DataLoader
NUM_WORKERS = 2
PIN_MEMORY  = True

# ==================== DEVICE ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== TRANSFORMS ====================
# ImageNet grayscale normalization stats
NORM_MEAN = [0.485]
NORM_STD  = [0.229]
