FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY audio audio
COPY runs/vision_v2/best_model.pth runs/vision_v2/best_model.pth
COPY runs/vision_v2/classes.json runs/vision_v2/classes.json
COPY runs/audio_v1/best_model.pth runs/audio_v1/best_model.pth
COPY runs/audio_v1/classes.json runs/audio_v1/classes.json
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "app/app.py"]
