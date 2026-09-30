"""Evaluate the deployed audio preprocessing on held-out recordings (no training)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.inference import BirdIdentifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=Path, default=Path('data/audio/test'))
    parser.add_argument('--output', type=Path, default=Path('runs/audio_v1/test_results.json'))
    args = parser.parse_args()
    identifier = BirdIdentifier()
    rows = []
    for species in sorted(args.test_dir.iterdir()):
        if not species.is_dir():
            continue
        for recording in sorted(species.iterdir()):
            if recording.suffix.lower() not in {'.mp3', '.wav', '.flac', '.ogg'}:
                continue
            try:
                predictions = identifier.audio(recording)
                rows.append({'file': f'{species.name}/{recording.name}', 'expected': species.name,
                            'predicted': predictions[0]['species'],
                            'top5_correct': species.name in [p['species'] for p in predictions]})
            except ValueError as error:
                rows.append({'file': f'{species.name}/{recording.name}', 'expected': species.name, 'error': str(error)})
        print(f'Evaluated {species.name}', flush=True)
    valid = [r for r in rows if 'predicted' in r]
    result = {'recordings': len(rows), 'evaluated': len(valid), 'skipped': len(rows) - len(valid),
              'top1_accuracy': sum(r['expected'] == r['predicted'] for r in valid) / max(1, len(valid)),
              'top5_accuracy': sum(r['top5_correct'] for r in valid) / max(1, len(valid)),
              'max_seconds': 30, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
