"""
Step 4: Wrap both models in a Gradio app and deploy to Hugging Face Spaces.

Local run:
    python app.py

Deploy to HF Spaces (free CPU tier is fine for inference):
    1. huggingface-cli login
    2. Create a Space (SDK: Gradio) at huggingface.co/new-space
    3. Copy this whole app/ folder's contents, plus your model checkpoints
    and classes.json files, into the Space repo
    4. git add . && git commit -m "deploy" && git push

Expected files alongside this script (copy them in before deploying):
    vision_model.pth, vision_classes.json
    audio_model.pth,  audio_classes.json
"""

import json
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

APP_DIR = Path(__file__).parent
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------- Vision model ----------
vision_classes = json.loads((APP_DIR / "vision_classes.json").read_text())
vision_model = models.resnet18(weights=None)
vision_model.fc = nn.Linear(vision_model.fc.in_features, len(vision_classes))
vision_model.load_state_dict(torch.load(APP_DIR / "vision_model.pth", map_location=device))
vision_model.to(device).eval()

vision_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(img: Image.Image):
    if img is None:
        return {}
    x = vision_tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(vision_model(x), dim=1)[0]
    top5 = torch.topk(probs, k=min(5, len(vision_classes)))
    return {vision_classes[i]: float(p) for p, i in zip(top5.values, top5.indices)}


# ---------- Audio model ----------
from importlib import import_module
import sys
sys.path.append(str(APP_DIR.parent / "audio"))
try:
    from train_audio_cnn import SmallAudioCNN
    from audio_to_melspec import clip_to_spectrogram_images
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

if AUDIO_AVAILABLE and (APP_DIR / "audio_model.pth").exists():
    audio_classes = json.loads((APP_DIR / "audio_classes.json").read_text())
    audio_model = SmallAudioCNN(len(audio_classes))
    audio_model.load_state_dict(torch.load(APP_DIR / "audio_model.pth", map_location=device))
    audio_model.to(device).eval()
    audio_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def predict_audio(audio_path):
    if not AUDIO_AVAILABLE or audio_path is None:
        return {}
    tmp_dir = Path("_tmp_app_spectrograms")
    tmp_dir.mkdir(exist_ok=True)
    n_saved = clip_to_spectrogram_images(Path(audio_path), tmp_dir, "clip")
    if n_saved == 0:
        return {}

    probs_sum = torch.zeros(len(audio_classes))
    n = 0
    for img_path in tmp_dir.glob("clip_*.png"):
        img = Image.open(img_path).convert("RGB")
        x = audio_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs_sum += torch.softmax(audio_model(x), dim=1).cpu()[0]
        n += 1
        img_path.unlink()

    if n == 0:
        return {}
    avg_probs = probs_sum / n
    top5 = torch.topk(avg_probs, k=min(5, len(audio_classes)))
    return {audio_classes[i]: float(p) for p, i in zip(top5.values, top5.indices)}


# ---------- UI ----------
with gr.Blocks(title="Bird ID: Image + Call") as demo:
    gr.Markdown("# Bird Species Identification\nUpload a photo or a bird call recording.")
    with gr.Tab("From photo"):
        img_input = gr.Image(type="pil", label="Bird photo")
        img_output = gr.Label(num_top_classes=5, label="Predicted species")
        img_input.change(predict_image, inputs=img_input, outputs=img_output)

    if AUDIO_AVAILABLE:
        with gr.Tab("From call"):
            audio_input = gr.Audio(type="filepath", label="Bird call recording")
            audio_output = gr.Label(num_top_classes=5, label="Predicted species")
            audio_input.change(predict_audio, inputs=audio_input, outputs=audio_output)

if __name__ == "__main__":
    demo.launch()