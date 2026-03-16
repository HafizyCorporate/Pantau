FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown untuk download dari Google Drive
RUN pip install gdown

# Pre-download EasyOCR model
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

RUN mkdir -p temp models

# Download YOLOv8n base model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    mv yolov8n.pt models/yolov8n.pt || true

# Download seatbelt model dari Google Drive
RUN gdown "https://drive.google.com/uc?id=1SALSjn9DEzddXYYOsOvm3bGKQmXeiqlX" -O models/seatbelt.pt

CMD ["python", "app.py"]
