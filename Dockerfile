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

RUN pip install gdown

RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

RUN mkdir -p temp models

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    mv yolov8n.pt models/yolov8n.pt || true

RUN gdown --fuzzy "https://drive.google.com/file/d/1SALSjn9DEzddXYYOsOvm3bGKQmXeiqlX/view?usp=drivesdk" -O models/seatbelt.pt

CMD ["python", "app.py"]
