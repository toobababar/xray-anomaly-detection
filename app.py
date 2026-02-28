# ============================================================
# app.py
# FastAPI application for Chest X-Ray Classification.
# Run locally:  uvicorn app:app --reload
# Run via Docker: docker run -p 8000:8000 chest-xray-classifier
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from config import MODEL_PATH, CLASS_NAMES
from model import load_model
from inference import predict


# ==================== APP SETUP ====================
app = FastAPI(
    title="Chest X-Ray Classification API",
    description="Classifies chest X-ray images into 5 respiratory disease categories.",
    version="1.0.0"
)

# Load model once at startup — not on every request
model = load_model(MODEL_PATH)


# ==================== ROUTES ====================
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "model":   "ResNet50 Chest X-Ray Classifier",
        "classes": CLASS_NAMES
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Classify a chest X-ray image.

    Args:
        file: uploaded image file (PNG, JPG, JPEG)

    Returns:
        predicted class, confidence score, and all class probabilities
    """
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG."
        )

    # Read and decode image
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {str(e)}"
        )

    # Run inference
    try:
        result = predict(image, model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    return JSONResponse(content={
        "filename":        file.filename,
        "predicted_class": result["predicted_class"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"]
    })