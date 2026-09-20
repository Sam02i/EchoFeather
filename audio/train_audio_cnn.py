"""
Step 2c: Train a small CNN on the mel-spectrogram images.

Same idea as the vision model (step 1), but from scratch and smaller:
spectrograms don't look like natural photos, so ImageNet pretraining
gives less benefit here, and a lighter net trains faster on this much data.

Input: data/audio/spectrograms/<species>/*.png  (from audio_to_melspec.py)

Usage:
    python train_audio_cnn.py --data_dir data/audio/spectrograms --epochs 20 \
        --out_dir runs/audio_v1
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None


class SmallAudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def per_species_accuracy(labels, preds, class_names):
    counts = {c: [0, 0] for c in class_names}
    for y, p in zip(labels, preds):
        c = class_names[y]
        counts[c][1] += 1
        if y == p:
            counts[c][0] += 1
    return {c: (correct / total if total else None) for c, (correct, total) in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out_dir", default="runs/audio_v1")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_ds = datasets.ImageFolder(args.data_dir, transform=tf)
    class_names = full_ds.classes
    (out_dir / "classes.json").write_text(json.dumps(class_names, indent=2))

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SmallAudioCNN(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"epoch {epoch} val", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_acc = val_correct / val_total

        print(f"[{epoch}/{args.epochs}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            per_species = per_species_accuracy(all_labels, all_preds, class_names)
            (out_dir / "per_species_accuracy.json").write_text(json.dumps(per_species, indent=2))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val accuracy: {best_acc:.3f}")
    print("Saved best_model.pth + per_species_accuracy.json in", out_dir)


if __name__ == "__main__":
    main()