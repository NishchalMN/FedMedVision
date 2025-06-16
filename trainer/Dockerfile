FROM python:3.10-slim

# Install system packages (for Pillow image support)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU only) and deps
RUN pip install --no-cache-dir \
    torch torchvision \
    scikit-learn requests pandas pillow tqdm

# Copy client script
WORKDIR /app
COPY trainer.py .

# Run the client when the container starts
CMD ["python", "trainer.py"]
