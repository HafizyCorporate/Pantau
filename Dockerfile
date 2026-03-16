FROM python:3.10-slim

WORKDIR /app

# Install dependencies sistem untuk OpenCV & PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Buat folder temp di dalam container
RUN mkdir -p temp

CMD ["python", "app.py"]
