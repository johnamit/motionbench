FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# scripts to run the app and utilities
COPY scripts ./scripts

# Clone the models from my Hugging Face repository
RUN git clone --depth 1 https://huggingface.co/johnamit/motionbench-models /tmp/motionbench-models \
    && mv /tmp/motionbench-models/models /app/models \
    && rm -rf /tmp/motionbench-models

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "scripts/app/motionbench.py", "--server.port=7860", "--server.address=0.0.0.0"]