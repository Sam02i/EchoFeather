"""Split recordings before creating spectrograms, so clips cannot leak across splits."""
import argparse
import random
from pathlib import Path
from audio_to_melspec import clip_to_spectrogram_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/audio/train'))
    parser.add_argument('--output', type=Path, default=Path('data/audio/spectrograms_split'))
    parser.add_argument('--seconds', type=int, default=15)
    args = parser.parse_args()
    for species in sorted(args.input.iterdir()):
        if not species.is_dir():
            continue
        recordings = sorted(p for p in species.iterdir() if p.suffix.lower() in {'.mp3', '.wav', '.flac', '.ogg'})
        random.Random(42).shuffle(recordings)
        if len(recordings) < 2:
            continue
        validation_count = max(1, round(len(recordings) * .2))
        count = 0
        for index, recording in enumerate(recordings):
            split = 'val' if index < validation_count else 'train'
            output = args.output / split / species.name
            output.mkdir(parents=True, exist_ok=True)
            if list(output.glob(f'{recording.stem}_*.png')):
                continue
            count += clip_to_spectrogram_images(recording, output, recording.stem, max_seconds=args.seconds)
        print(f'{species.name}: {len(recordings)} recordings, {count} new spectrograms', flush=True)


if __name__ == '__main__':
    main()
