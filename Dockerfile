# ============================================================
# Dockerfile
# Containerizes the Chest X-Ray Classification FastAPI app.
# ============================================================

# Use slim Python image for smaller container size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and PIL
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first (layer caching — only reinstalls if these change)
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY config.py .
COPY model.py .
COPY inference.py .
COPY app.py .

# Copy trained model weights
COPY models/ ./models/

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
