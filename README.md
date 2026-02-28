# Chest X-Ray Anomaly Detection

A deep learning API for classifying chest X-ray images into 5 respiratory disease categories using a fine-tuned ResNet50 model trained on the Chest X-Ray Dataset for Respiratory Disease Classification dataset available on Harvard Dataverse.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toobababar/xray-anomaly-detection/blob/main/main.ipynb)

## Classes

| Label | Disease |
|-------|---------|
| 0 | COVID-19 |
| 1 | Lung Opacity |
| 2 | Normal |
| 3 | Viral Pneumonia |
| 4 | Tuberculosis |

## Project Structure

```
xray-anomaly-detection/
    data/
        README.md               # dataset download instructions
    models/
        resnet50_chestxray_v1.pth   # trained model weights (Git LFS)
    scripts/
        download_data.py        # automated dataset download
    app.py                      # FastAPI application
    config.py                   # centralized configuration
    dataset.py                  # dataset class and dataloader
    Dockerfile                  # container definition
    evaluate.py                 # model evaluation and metrics
    export_onnx.py              # export model to ONNX format
    inference.py                # preprocessing and prediction
    model.py                    # ResNet50 architecture
    train.py                    # training script
    pyproject.toml              # dependencies managed with uv
    main.ipynb                  # full experiment notebook (Colab)
```

---

## For Users — Run with Docker

Just want to classify chest X-rays? No setup required beyond Docker.

### Pull and run the container

```bash
docker pull toobababar/chest-xray-classifier
docker run -p 8000:8000 toobababar/chest-xray-classifier
```

### API is now available at

```
http://localhost:8000
http://localhost:8000/docs      # interactive Swagger UI
```

### Classify an X-ray image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.png"
```

### Example response

```json
{
  "filename": "chest_xray.png",
  "predicted_class": "normal",
  "confidence": 0.9821,
  "probabilities": {
    "covid": 0.0021,
    "lung_opacity": 0.0089,
    "normal": 0.9821,
    "viral_pneumonia": 0.0043,
    "tuberculosis": 0.0026
  }
}
```

---

## For Developers — Build on Top

Want to retrain, fine-tune, extend the API, or export to ONNX?

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- CUDA-compatible GPU recommended for training (Google Colab works)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/xray-anomaly-detection
cd xray-anomaly-detection

# Install dependencies
uv pip install -r pyproject.toml
```

### Download dataset

```bash
python scripts/download_data.py
```

See `data/README.md` for manual download instructions.

### Train

```bash
python train.py
```

Training curves are saved to `training_curves.png`.

### Evaluate

```bash
python evaluate.py
```

Outputs classification report and confusion matrix saved to `confusion_matrix.png`.

### Export to ONNX

For deployment on non-PyTorch platforms:

```bash
python export_onnx.py
```

### Run API locally

```bash
uvicorn app:app --reload
```

### Build Docker image locally

```bash
docker build -t chest-xray-classifier .
docker run -p 8000:8000 chest-xray-classifier
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | ResNet50 |
| Pretrained on | ImageNet |
| Input | Grayscale X-ray (224x224) |
| Output | 5-class softmax |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 50 |
| Best Val Accuracy | ~82% |

---

## Dataset

Chest X-Ray Dataset for Respiratory Disease Classification
This project uses the Chest X-Ray Dataset for Respiratory Disease Classification, a publicly available multi-class chest X-ray dataset containing five categories:
COVID-19
Pneumonia
Tuberculosis
Lung Opacity
Normal

The dataset is hosted on Harvard Dataverse and was published in 2021.

DOI: https://doi.org/10.7910/DVN/WNQ3GI

Version Used: V5

⚠️ This repository does not redistribute the dataset.
Please download it directly from the official source:
https://doi.org/10.7910/DVN/WNQ3GI

---
## Citation

If you use this repository or the dataset in your research, please cite:

Basu, A., Das, S., Ghosh, S., Mullick, S., Gupta, A., & Das, S. (2021). Chest X-Ray Dataset for Respiratory Disease Classification (V5) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/WNQ3GI

```bibtex
@data{DVN/WNQ3GI_2021,
author = {Basu, Arkaprabha and Das, Sourav and Ghosh, Susmita and Mullick, Sankha and Gupta, Avisek and Das, Swagatam},
publisher = {Harvard Dataverse},
title = {{Chest X-Ray Dataset for Respiratory Disease Classification}},
year = {2021},
version = {V5},
doi = {10.7910/DVN/WNQ3GI},
url = {https://doi.org/10.7910/DVN/WNQ3GI}
}
```
---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you would like to change.
