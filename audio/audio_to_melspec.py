"""
Step 2b: Turn audio classification into image classification.

This is the trick worth understanding, not just running: a mel-spectrogram
is a 2D image where x = time, y = frequency (on a perceptual/mel scale), and
pixel intensity = loudness at that time+frequency. Once a bird call looks
like a picture, the same CNN architecture and training loop from the vision
model applies directly.

Long recordings get sliced into fixed-length chunks (default 5s) so every
training example is the same size and a single clip yields multiple samples.

Input:  data/audio/raw/<species>/*.mp3
Output: data/audio/spectrograms/<species>/<clip_id>_<chunk_idx>.png

Usage:
    python audio_to_melspec.py --in_dir data/audio/raw --out_dir data/audio/spectrograms
"""

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # no display backend needed
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 22050
CHUNK_SECONDS = 5
N_MELS = 128


def clip_to_spectrogram_images(audio_path, out_dir, clip_id):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"    could not load {audio_path.name}: {e}")
        return 0

    chunk_len = CHUNK_SECONDS * sr
    n_chunks = max(1, len(y) // chunk_len)
    saved = 0

    for i in range(n_chunks):
        chunk = y[i * chunk_len:(i + 1) * chunk_len]
        if len(chunk) < chunk_len * 0.6:  # skip near-empty trailing chunk
            continue

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)  # -> ~224x224px, matches vision input
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        librosa.display.specshow(mel_db, sr=sr, ax=ax, cmap="magma")
        out_path = out_dir / f"{clip_id}_{i}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/audio/raw")
    ap.add_argument("--out_dir", default="data/audio/spectrograms")
    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)

    species_dirs = [d for d in in_root.iterdir() if d.is_dir()]
    for species_dir in species_dirs:
        out_dir = out_root / species_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        clips = list(species_dir.glob("*.mp3")) + list(species_dir.glob("*.wav"))
        total_saved = 0
        for clip in tqdm(clips, desc=species_dir.name):
            total_saved += clip_to_spectrogram_images(clip, out_dir, clip.stem)
        print(f"[{species_dir.name}] {len(clips)} clips -> {total_saved} spectrogram images")


if __name__ == "__main__":
    main()
