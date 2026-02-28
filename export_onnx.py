# ============================================================
# export_onnx.py
# Export trained ResNet50 model to ONNX format for deployment.
# ONNX is framework-agnostic and faster at inference time.
# Run: python export_onnx.py
# ============================================================

import torch
from config import DEVICE, MODEL_PATH, ONNX_PATH, IMG_SIZE
from model import load_model


def export_to_onnx(model_path=MODEL_PATH, onnx_path=ONNX_PATH):
    """
    Export trained model to ONNX format.

    Args:
        model_path: path to saved .pth weights file
        onnx_path:  path to save the .onnx file
    """
    # Load trained model
    model = load_model(model_path)
    model.eval()

    # Dummy input matching model's expected input shape
    # (batch=1, channels=1, height=IMG_SIZE, width=IMG_SIZE)
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,          # store trained weights in the model
        opset_version=14,            # ONNX opset version
        do_constant_folding=True,    # optimize constant expressions
        input_names=["input"],       # name for the input node
        output_names=["output"],     # name for the output node
        dynamic_axes={
            "input":  {0: "batch_size"},   # allow variable batch size
            "output": {0: "batch_size"}
        }
    )

    print(f"Model exported to ONNX: {onnx_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed.")
    except ImportError:
        print("onnx package not installed — skipping verification.")
        print("Install with: pip install onnx")


if __name__ == "__main__":
    export_to_onnx()
