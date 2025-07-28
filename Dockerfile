# Use slim python base
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for OCR and PDF tools)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Ensure input/output folders exist
RUN mkdir -p /app/input /app/output

# Run the script
CMD ["python", "main.py"]


# to build cmd -->  docker build --platform linux/amd64 -t mysolution:abc123 .
# to run cmd -->   docker run --rm -v C:/DOCKER_SAMPLE/ADOBE_1B/app/input:/app/input -v C:/DOCKER_SAMPLE/ADOBE_1B/app/output:/app/output --network none mysolution:abc123

