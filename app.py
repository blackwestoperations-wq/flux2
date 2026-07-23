import os
import time
import threading
from io import BytesIO

# Must be set BEFORE importing gradio
os.environ["GRADIO_MCP_SERVER"] = "True"

import torch
from huggingface_hub import login
from diffusers import Flux2Pipeline
import gradio as gr
from fastapi import FastAPI
from PIL import Image

# ─── Presets (identical to your Modal version) ───────────────────

RESOLUTION_PRESETS = {
    "1:1 Square (1024×1024)": (1024, 1024),
    "16:9 Landscape (1360×768)": (1360, 768),
    "9:16 Portrait (768×1360)": (768, 1360),
    "4:3 Standard (1152×896)": (1152, 896),
    "3:4 Portrait (896×1152)": (896, 1152),
    "3:2 Photo (1216×832)": (1216, 832),
    "2:3 Portrait Photo (832×1216)": (832, 1216),
    "21:9 Ultrawide (1536×640)": (1536, 640),
    "2K HD (1920×1080)": (1920, 1080),
    "2K Vertical (1080×1920)": (1080, 1920),
}

QUALITY_PRESETS = {
    "⚡ Fast (20 steps)": 20,
    "🔄 Balanced (28 steps)": 28,
    "✨ Quality (35 steps)": 35,
    "🎨 Maximum (50 steps)": 50,
}

# ─── Lazy Model Loading ──────────────────────────────────────────
# Load the model in a background thread so the Gradio server
# starts immediately and passes Koyeb health checks.

pipe = None
pipe_lock = threading.Lock()
model_status = "idle"  # idle → loading → ready → error


def load_model():
    global pipe, model_status
    try:
        model_status = "loading"
        print("⏳ Loading FLUX.2-dev...")

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Logged in to HuggingFace")

        repo_id = "black-forest-labs/FLUX.2-dev"
        pipe = Flux2Pipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        pipe.to("cuda")

        try:
            pipe.transformer.fuse_qkv_projections()
            pipe.vae.fuse_qkv_projections()
            print("✅ QKV projections fused")
        except AttributeError:
            pass

        model_status = "ready"
        print("✅ Model loaded!")
    except Exception as e:
        model_status = "error"
        print(f"❌ Model load failed: {e}")


def get_pipe():
    """Block until the model is ready (called from generate function)."""
    global model_status
    if pipe is None and model_status != "loading":
        load_model()
    while model_status == "loading":
        time.sleep(1)
    if model_status == "error":
        raise RuntimeError("Model failed to load. Check container logs.")
    return pipe


# Start loading in background immediately
threading.Thread(target=load_model, daemon=True).start()

# ─── Generation Function ─────────────────────────────────────────


def generate_flux_image(
    prompt: str,
    aspect_ratio: str = "1:1 Square (1024×1024)",
    quality_preset: str = "🔄 Balanced (28 steps)",
    guidance: str = "3.5",
    seed: str = "42",
    progress: gr.Progress() = None,
):
    """
    Generate high-quality images using Flux.2-Dev on Koyeb GPU.

    Args:
        prompt: Detailed text description of the image.
        aspect_ratio: Image aspect ratio preset.
        quality_preset: Quality/speed preset.
        guidance: Guidance scale (1.0-10.0).
        seed: Random seed for reproducibility.
    """
    sd = int(seed)
    g = float(guidance)
    s = QUALITY_PRESETS.get(quality_preset, 28)
    w, h = RESOLUTION_PRESETS.get(aspect_ratio, (1024, 1024))

    if model_status == "loading" and progress:
        progress(0.1, desc="Waiting for model to finish loading...")

    p = get_pipe()

    if progress:
        progress(0.3, desc=f"Generating ({s} steps)...")
    print(f"🎨 Generating: {prompt}")
    start_time = time.time()

    generator = torch.Generator(device="cuda").manual_seed(sd)
    out = p(
        prompt=prompt,
        width=w,
        height=h,
        num_inference_steps=s,
        guidance_scale=g,
        generator=generator,
    ).images[0]

    elapsed = time.time() - start_time
    print(f"✅ Generated in {elapsed:.1f}s")

    if progress:
        progress(1.0, desc=f"Done! ({elapsed:.1f}s)")

    return out


# ─── Gradio Interface ─────────────────────────────────────────────

demo = gr.Interface(
    fn=generate_flux_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            lines=3,
            placeholder="A cat holding a sign that says 'Hello FLUX.2'",
        ),
        gr.Dropdown(
            choices=list(RESOLUTION_PRESETS.keys()),
            value="1:1 Square (1024×1024)",
            label="Aspect Ratio",
        ),
        gr.Dropdown(
            choices=list(QUALITY_PRESETS.keys()),
            value="🔄 Balanced (28 steps)",
            label="Quality",
        ),
        gr.Slider(1.0, 10.0, 3.5, step=0.5, label="Guidance Scale"),
        gr.Number(42, label="Seed"),
    ],
    outputs=gr.Image(label="Result"),
    title="🎨 FLUX.2-Dev on Koyeb",
    description=(
        "Generate images with FLUX.2-Dev on serverless GPU. "
        "MCP enabled for AI agents."
    ),
    api_name="generate",
)

demo.queue()

# ─── Launch ──────────────────────────────────────────────────────
# Koyeb sets PORT env var; bind to 0.0.0.0 for external access.

port = int(os.environ.get("PORT", 8000))
demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
