# Bird ID Pipeline: Image + Call

Four-step pipeline matching the plan: fine-tune the vision model properly,
build the audio side from spectrograms, benchmark against BirdNET, then
deploy both as one Gradio app.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Vision model (2-3 days)

Data layout expected: `data/images/train/<species>/*.jpg` and `data/images/val/<species>/*.jpg`

```bash
python vision/finetune_vision.py \
    --data_dir data/images \
    --epochs 15 \
    --unfreeze_from layer3 \
    --pretrained_ckpt path/to/your/existing/head_only_model.pth \
    --out_dir runs/vision_v2
```

Outputs `runs/vision_v2/best_model.pth`, `per_species_accuracy.json`, and
`top_confused_pairs.json` — the confused-pairs file is what goes in your
README, not just the top-line accuracy number.

## Step 2 — Audio side (3-4 days)

```bash
# 2a. pull Indian bird calls (put target species, one per line, in species.txt)
python audio/download_xenocanto.py --species_file species.txt \
    --out_dir data/audio/raw --country India --max_per_species 40

# 2b. convert to mel-spectrogram images (5s chunks)
python audio/audio_to_melspec.py --in_dir data/audio/raw --out_dir data/audio/spectrograms

# 2c. train a small CNN on the spectrogram images
python audio/train_audio_cnn.py --data_dir data/audio/spectrograms \
    --epochs 20 --out_dir runs/audio_v1
```

## Step 3 — Benchmark against BirdNET (1-2 days)

Set aside a held-out test set at `data/audio/test/<species>/*.mp3` that wasn't
used in step 2c, then:

```bash
conda create -n birdnet python=3.11 -y
conda activate birdnet
pip install -r requirements-birdnet.txt

python audio/benchmark_birdnet.py \
    --test_dir data/audio/test \
    --your_model runs/audio_v1/best_model.pth \
    --classes runs/audio_v1/classes.json \
    --out_csv runs/benchmark_results.csv

conda deactivate  # back to your main venv for step 4
```

TensorFlow (which `birdnetlib` needs) doesn't ship wheels for Python 3.14 yet,
so this step needs its own older-Python environment — that's why it's a
separate requirements file instead of being in the main `requirements.txt`.

Report the result either way — a documented loss against a 6,000-species
model trained on far more data is a legitimate, interesting finding.

## Step 4 — Deploy (1 day)

```bash
cp runs/vision_v2/best_model.pth app/vision_model.pth
cp runs/vision_v2/classes.json app/vision_classes.json
cp runs/audio_v1/best_model.pth app/audio_model.pth
cp runs/audio_v1/classes.json app/audio_classes.json

cd app && python app.py   # sanity check locally
```

Then push `app/` (plus the four copied files and `requirements.txt`) to a new
Hugging Face Space (SDK: Gradio) for a free, live link:

```bash
huggingface-cli login
# create the Space at huggingface.co/new-space, then:
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cp app/* audio/train_audio_cnn.py audio/audio_to_melspec.py <cloned-space-dir>/
cp requirements.txt <cloned-space-dir>/
cd <cloned-space-dir> && git add . && git commit -m "deploy" && git push
```

Only add the live link to your resume once this step is done and working.
