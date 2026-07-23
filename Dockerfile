FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Environment ---
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_HUB_CACHE=/cache
ENV GRADIO_MCP_SERVER=True
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Python dependencies ---
RUN pip install --no-cache-dir \
    "invisible_watermark>=0.2.0" \
    "huggingface_hub" \
    "safetensors" \
    "sentencepiece" \
    "numpy<2" \
    "transformers==4.57.3" \
    "diffusers>=0.35.0" \
    "peft" \
    "accelerate" \
    "gradio[mcp]>=5.0.0" \
    "fastapi[standard]" \
    "pillow"

# --- App code ---
COPY app.py .

# Koyeb sets PORT env var; we default to 8000
ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
