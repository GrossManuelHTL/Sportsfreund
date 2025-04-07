FROM arm64v8/python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    libasound2-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
