"""
Step 3: Benchmark your audio model against BirdNET.

Uses the open-source `birdnetlib` package (wraps the BirdNET-Analyzer model)
as the baseline. Runs both your model and BirdNET on the same held-out test
clips and reports where each wins -- this comparison is the most interesting
sentence in your README, same spirit as the cache-eviction project. Report
a loss honestly if you get one; explaining *why* is worth more than hiding it.

pip install birdnetlib

Expected test set layout (ground-truth labelled):
    data/audio/test/<species>/*.mp3

Usage:
    python benchmark_birdnet.py --test_dir data/audio/test \
        --your_model runs/audio_v1/best_model.pth \
        --classes runs/audio_v1/classes.json \
        --out_csv runs/benchmark_results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from train_audio_cnn import SmallAudioCNN  # reuse the model class
from audio_to_melspec import clip_to_spectrogram_images

try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    _import_error = None
except ImportError as e:
    Analyzer = None
    _import_error = e  # keep the real reason -- usually a missing TF Lite runtime, not a missing package


def load_your_model(ckpt_path, classes_path, device):
    class_names = json.loads(Path(classes_path).read_text())
    model = SmallAudioCNN(len(class_names))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, class_names


def predict_with_your_model(model, class_names, audio_path, device, tmp_dir):
    tmp_dir.mkdir(exist_ok=True)
    n_saved = clip_to_spectrogram_images(audio_path, tmp_dir, "tmp")
    if n_saved == 0:
        return None, 0.0

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # average predictions across all chunks from this clip
    probs_sum = torch.zeros(len(class_names))
    n = 0
    for img_path in tmp_dir.glob("tmp_*.png"):
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu()[0]
        probs_sum += probs
        n += 1
        img_path.unlink()

    if n == 0:
        return None, 0.0
    avg_probs = probs_sum / n
    top_idx = avg_probs.argmax().item()
    return class_names[top_idx], avg_probs[top_idx].item()


def predict_with_birdnet(analyzer, audio_path):
    rec = Recording(analyzer, str(audio_path), min_conf=0.1)
    rec.analyze()
    if not rec.detections:
        return None, 0.0
    top = max(rec.detections, key=lambda d: d["confidence"])
    return top["common_name"], top["confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--your_model", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out_csv", default="runs/benchmark_results.csv")
    args = ap.parse_args()

    if Analyzer is None:
        raise SystemExit(
            f"Could not import birdnetlib's Analyzer. Real error: {_import_error}\n"
            "This is almost always a missing/incompatible TensorFlow, not a missing birdnetlib -- "
            "`pip install birdnetlib` alone is not enough. Run: pip install tensorflow\n"
            "(On Apple Silicon, plain `tensorflow` from PyPI now works natively -- no need for "
            "tensorflow-macos separately on recent versions.)"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model, class_names = load_your_model(args.your_model, args.classes, device)
    analyzer = Analyzer()

    tmp_dir = Path("_tmp_benchmark_spectrograms")
    test_dir = Path(args.test_dir)

    rows = []
    your_correct, birdnet_correct, total = 0, 0, 0

    for species_dir in [d for d in test_dir.iterdir() if d.is_dir()]:
        true_species = species_dir.name.replace("_", " ")
        for clip in list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav")):
            your_pred, your_conf = predict_with_your_model(model, class_names, clip, device, tmp_dir)
            bn_pred, bn_conf = predict_with_birdnet(analyzer, clip)

            your_hit = your_pred == true_species
            # BirdNET returns common names; exact-match comparison is approximate --
            # do a substring check as a looser fallback for naming mismatches
            bn_hit = bn_pred is not None and (
                bn_pred.lower() == true_species.lower() or true_species.lower() in bn_pred.lower()
            )

            your_correct += int(your_hit)
            birdnet_correct += int(bn_hit)
            total += 1

            rows.append({
                "clip": clip.name,
                "true_species": true_species,
                "your_model_pred": your_pred,
                "your_model_conf": round(your_conf, 3),
                "your_model_correct": your_hit,
                "birdnet_pred": bn_pred,
                "birdnet_conf": round(bn_conf, 3),
                "birdnet_correct": bn_hit,
            })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    your_acc = your_correct / total if total else 0
    bn_acc = birdnet_correct / total if total else 0
    print(f"\nTested on {total} clips")
    print(f"Your model accuracy:  {your_acc:.3f}")
    print(f"BirdNET accuracy:     {bn_acc:.3f}")
    if your_acc > bn_acc:
        print(f"-> Your model beats BirdNET by {your_acc - bn_acc:.3f} on this test set "
            f"(smaller species list + region-specific training data likely why).")
    elif your_acc < bn_acc:
        print(f"-> BirdNET beats your model by {bn_acc - your_acc:.3f} "
            f"(expected: it's trained on ~6,000 species with far more data per class). "
            f"Report this honestly -- explain the gap, don't hide it.")
    else:
        print("-> Dead even. Worth digging into per-species results to see where each wins.")
    print(f"Full per-clip results: {args.out_csv}")


if __name__ == "__main__":
    main()