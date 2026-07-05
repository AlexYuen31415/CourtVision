FROM python:3.11-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run gunicorn binding to the port Render gives us
CMD gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT