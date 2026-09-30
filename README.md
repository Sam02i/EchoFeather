# Samiksha — Bird identification

A plain HTML/CSS/JavaScript website with a small FastAPI backend. The green
mosaic flips tile by tile, and the same app identifies birds from photos or calls.
No frontend build tools or JavaScript framework are needed.

## Run locally

From this project folder:

```bash
source venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

Open **http://127.0.0.1:8000**. If you do not already have a virtual environment,
create one first with `python3 -m venv venv`.

- **Home:** interactive mosaic with individual tile flips.
- **Bird:** photo upload, preview, and five suggested species.
- **Audio:** recording upload, playback, and five suggested species.
- **Collection:** searchable species catalog, with photo/audio support labels.
- **About:** how the models work and what their scores mean.

Use `PORT=8001 python app/app.py` to change the port. By default the server only
listens on your computer. Set `HOST=0.0.0.0` when deploying to a server/container.

## Simple code map

```text
app/
  app.py                 # HTTP routes, upload validation, website serving
  inference.py           # checkpoint loading and photo/audio predictions
  static/
    index.html           # shared header and page container
    style.css            # layout, colors, responsive styles, tile animation
    app.js               # pages, uploads, search, result rendering
    mosaic.js            # original mapped tile-flip interaction
    assets/              # the two original mosaic images
    species/             # local examples for the species catalog
vision/finetune_vision.py # existing photo training pipeline
audio/                  # recording preparation, CNN training, evaluation
tests/test_app.py       # API and error-path checks
```

The frontend calls `/api/status`, `/api/species`, `/api/predict/image`, and
`/api/predict/audio`. The API and website share one origin, so no separate
frontend server or CORS configuration is required. Interactive API docs are at
`/docs`.

## Models and results

The app automatically uses:

```text
runs/vision_v2/best_model.pth
runs/vision_v2/classes.json
runs/audio_v1/best_model.pth
runs/audio_v1/classes.json
```

The existing photo model supports **20 species**. Its existing per-species
validation results are in `runs/vision_v2/per_species_accuracy.json`.

The audio CNN was trained on the real recordings already in this repository:
**19 species**, 12 epochs, and 1,247 spectrograms. Recordings were split before
spectrogram generation so chunks from one recording stay in the same split.
The best validation accuracy was **36.7%**. On the separate local test folder,
it achieved **30.0% top-1** and **66.25% top-5** accuracy on **80 readable
recordings**; one recording was too short. Full results are in
`runs/audio_v1/test_results.json`. This is an experimental baseline, not a
reliable field identification system. More balanced recordings and further
model work are needed to improve it. No BirdNET comparison has been run here.

Scores are softmax model outputs, not calibrated probabilities of correctness.
Both models can confidently misidentify a bird outside their supported species.
The synthetic checkpoints under `_smoke_test` are never used by the app.

For alternate checkpoints, set `VISION_MODEL`, `VISION_CLASSES`, `AUDIO_MODEL`,
and `AUDIO_CLASSES` to your file paths. Alternatively place `vision_model.pth`,
`vision_classes.json`, `audio_model.pth`, and `audio_classes.json` in `app/`.
Restart the server after replacing an already-loaded checkpoint. Models load
once on first use. Missing models show a clear unavailable state.

## Train and evaluate audio

Keep the held-out recordings in `data/audio/test` separate from training.

```bash
python audio/prepare_audio.py --input data/audio/train
python audio/train_audio_cnn.py \
  --data_dir data/audio/spectrograms_split --epochs 12 --workers 0 \
  --out_dir runs/audio_v1
python audio/evaluate_audio.py --test-dir data/audio/test
```

Preparation uses up to the first 15 seconds of each training recording and
creates five-second spectrograms. Delete or choose a different generated output
folder when changing preparation settings; existing spectrograms are reused.
The training script supports `train/` and `val/` folders. The legacy flat
species-folder layout is also accepted, but a recording-level split is preferred.
The app averages predictions over up to the first 30 seconds of an upload.

The optional `audio/benchmark_birdnet.py` script and `requirements-birdnet.txt`
are retained from the original pipeline. BirdNET needs its own compatible
Python/TensorFlow environment; it is not needed to run this website.

## Train the photo model

```bash
python vision/finetune_vision.py --data_dir data/images \
  --epochs 15 --unfreeze_from layer3 --out_dir runs/vision_v2
```

Use the existing data preparation/download scripts when adding training data.
Your existing changes in `prepare_splits.py` have been preserved.

## Verification

```bash
python -m unittest discover -s tests -v
node --check app/static/app.js
node --check app/static/mosaic.js
```

Tests cover real photo inference, page routing, the species catalog, missing
models, invalid uploads, and file-size limits. Photo and audio uploads were
also exercised through the browser. The original `test_pipeline_smoke.py`
remains available for synthetic training checks; it replaces `_smoke_test`
when run, so preserve anything you need from that folder first.

Uploads are limited to 25 MB and images to 20 megapixels. Photos are decoded
in memory. Audio uses a unique temporary directory for each request; recordings
and spectrograms are deleted when processing finishes. An inference lock prevents
concurrent audio requests from sharing plotting state. No uploads are saved to
training data or a user collection.

## Deployment

The former hosted mosaic is a static demo. This integrated app needs a Python
server for PyTorch inference; it cannot run solely on static website hosting.
A Dockerfile is included for a Python-capable host:

```bash
docker build -t samiksha-bird-id .
docker run --rm -p 8000:8000 samiksha-bird-id
```

The image includes the two trained checkpoints currently under `runs/` and the
frontend, but excludes raw datasets, the virtual environment and smoke outputs.
Docker deployment has not been executed as part of the local verification.
Model weights are ignored by Git; include them separately when deploying from
a fresh checkout. The integrated app has been run locally, not published to a
remote Python host.
