"""
Smoke test: runs every stage of the pipeline end-to-end on tiny synthetic
data (fake images, fake audio) so you catch bugs in minutes, not after a
multi-day real training run fails halfway through.

This does NOT check whether the model learns anything real -- it only
checks that data flows through each script without crashing and produces
the expected output files. Run this first, every time you change a script.

Usage:
    python test_pipeline_smoke.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST_ROOT = Path("_smoke_test")
SPECIES = ["species_a", "species_b", "species_c"]


def make_fake_images(root, n_per_species=8):
    for split in ["train", "val"]:
        for sp in SPECIES:
            d = root / "images" / split / sp
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_species):
                arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i}.jpg")


def make_fake_audio(root, n_per_species=4):
    import soundfile as sf
    for sp in SPECIES:
        d = root / "audio" / "raw" / sp
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_species):
            y = np.random.randn(22050 * 6).astype(np.float32) * 0.1  # 6s of noise
            sf.write(d / f"{i}.wav", y, 22050)


def run(cmd, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {label}")
        sys.exit(1)
    print(f"OK: {label}")


def main():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir()

    print("Generating synthetic data...")
    make_fake_images(TEST_ROOT)
    make_fake_audio(TEST_ROOT)

    # --- vision stage ---
    run(
        f"python vision/finetune_vision.py "
        f"--data_dir {TEST_ROOT}/images --epochs 1 --batch_size 4 "
        f"--out_dir {TEST_ROOT}/runs/vision",
        "Vision fine-tuning (1 epoch, tiny data)",
    )
    assert (TEST_ROOT / "runs/vision/best_model.pth").exists()
    assert (TEST_ROOT / "runs/vision/per_species_accuracy.json").exists()

    # --- audio: spectrogram conversion ---
    run(
        f"python audio/audio_to_melspec.py "
        f"--in_dir {TEST_ROOT}/audio/raw --out_dir {TEST_ROOT}/audio/spectrograms",
        "Mel-spectrogram conversion",
    )
    n_pngs = len(list((TEST_ROOT / "audio/spectrograms").rglob("*.png")))
    assert n_pngs > 0, "no spectrogram images were produced"
    print(f"  -> {n_pngs} spectrogram images generated")

    # --- audio: training ---
    run(
        f"python audio/train_audio_cnn.py "
        f"--data_dir {TEST_ROOT}/audio/spectrograms --epochs 1 --batch_size 4 "
        f"--val_split 0.3 --out_dir {TEST_ROOT}/runs/audio",
        "Audio CNN training (1 epoch, tiny data)",
    )
    assert (TEST_ROOT / "runs/audio/best_model.pth").exists()

    print("\nAll stages ran successfully on synthetic data.")
    print(f"Test artifacts left in {TEST_ROOT}/ for inspection (delete when done).")
    print("\nNOT covered by this smoke test (need real network calls, test separately):")
    print("  - download_xenocanto.py  -> hits the live Xeno-Canto API")
    print("  - benchmark_birdnet.py   -> downloads/runs the real BirdNET model")
    print("  - app/app.py             -> needs real trained checkpoints to load")


if __name__ == "__main__":
    main()
