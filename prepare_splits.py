"""
Turns a flat "one folder per species" collection into the train/val (images)
or train/test (audio) split the other scripts expect. Run this once you've
gathered real images/audio into:

    data/images_raw/<species>/*.jpg
    data/audio_raw/<species>/*.mp3

Splitting happens at the FILE level, before spectrogram conversion for audio --
that's what keeps your BirdNET benchmark test set honestly held out (chunks
from the same recording never leak between train and test).

Usage:
    python prepare_splits.py --kind images --in_dir data/images_raw --out_dir data/images --val_frac 0.2
    python prepare_splits.py --kind audio  --in_dir data/audio_raw  --out_dir data/audio  --val_frac 0.15 --test_frac 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTS = {".mp3", ".wav"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["images", "audio"], required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.0,
                    help="only used for --kind audio; held out for benchmark_birdnet.py")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    exts = IMAGE_EXTS if args.kind == "images" else AUDIO_EXTS
    in_root, out_root = Path(args.in_dir), Path(args.out_dir)

    for species_dir in sorted(d for d in in_root.iterdir() if d.is_dir()):
        files = [f for f in species_dir.iterdir() if f.suffix.lower() in exts]
        random.shuffle(files)
        if len(files) < 5:
            print(f"[{species_dir.name}] only {len(files)} files -- skipping, need more data")
            continue

        n_val = max(1, int(len(files) * args.val_frac))
        n_test = int(len(files) * args.test_frac)
        val_files = files[:n_val]
        test_files = files[n_val:n_val + n_test]
        train_files = files[n_val + n_test:]

        if args.kind == "images":
            splits = {"train": train_files, "val": val_files}
        else:
            splits = {"train": train_files + val_files, "test": test_files} if test_files \
                else {"train": train_files + val_files}

        for split_name, split_files in splits.items():
            dest = out_root / split_name / species_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest / f.name)

        counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"[{species_dir.name}] {len(files)} files -> {counts}")


if __name__ == "__main__":
    main()
