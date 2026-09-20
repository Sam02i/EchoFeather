"""
Step 1: Make the vision model actually good.

Adds the stuff you skipped the first time around:
  - real data augmentation (random crop, flip, color jitter) to handle the
    angle/lighting/background variation the job posting calls out
  - unfreezing more layers for a deeper fine-tune, not just the head
  - per-species accuracy + a confusion matrix, so you can honestly report
    *which* species get confused with which, not just a top-line number

Expected data layout (standard torchvision ImageFolder format):

    data/images/train/<species_name>/*.jpg
    data/images/val/<species_name>/*.jpg

Usage:
    python finetune_vision.py --data_dir data/images --epochs 15 \
        --unfreeze_from layer3 --out_dir runs/vision_v2
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None


def build_transforms(img_size=224):
    # Augmentation only on train: this is the part that was skipped.
    # RandomResizedCrop -> handles framing/distance variation
    # RandomHorizontalFlip -> birds face either direction in the wild
    # ColorJitter -> handles the lighting problem (overcast vs harsh sun)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def unfreeze_from(model, layer_name):
    """
    Unfreeze everything from `layer_name` onward (inclusive) plus the fc head.
    For resnet18/34/50, valid layer_name values: layer1, layer2, layer3, layer4.
    Unfreezing from layer3 onward is a good default: it fine-tunes the
    mid-to-high level feature detectors without destabilizing the low-level
    edge/texture filters learned from ImageNet.
    """
    unfreeze = False
    for name, child in model.named_children():
        if name == layer_name:
            unfreeze = True
        if unfreeze or name == "fc":
            for p in child.parameters():
                p.requires_grad = True
        else:
            for p in child.parameters():
                p.requires_grad = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    per_species = per_species_accuracy(all_labels, all_preds, class_names)
    confused_pairs = top_confused_pairs(all_labels, all_preds, class_names)
    return running_loss / total, correct / total, per_species, confused_pairs


def per_species_accuracy(labels, preds, class_names):
    counts = {c: [0, 0] for c in class_names}  # [correct, total]
    for y, p in zip(labels, preds):
        c = class_names[y]
        counts[c][1] += 1
        if y == p:
            counts[c][0] += 1
    return {
        c: (correct / total if total else None)
        for c, (correct, total) in counts.items()
    }


def top_confused_pairs(labels, preds, class_names, top_k=10):
    """Which species get mistaken for which, ranked by frequency."""
    if confusion_matrix is None:
        return []
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i][j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with train/ and val/ subfolders")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze_from", default="layer3",
                     choices=["layer1", "layer2", "layer3", "layer4", "fc"])
    ap.add_argument("--pretrained_ckpt", default=None,
                     help="optional: path to your existing head-only fine-tuned checkpoint")
    ap.add_argument("--out_dir", default="runs/vision_v2")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms()
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)
    class_names = train_ds.classes
    (out_dir / "classes.json").write_text(json.dumps(class_names, indent=2))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    if args.pretrained_ckpt:
        state = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded existing checkpoint: {args.pretrained_ckpt}")

    unfreeze_from(model, args.unfreeze_from)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable):,} params "
          f"(unfrozen from {args.unfreeze_from} onward)")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, per_species, confused = evaluate(model, val_loader, criterion, device, class_names)
        scheduler.step()

        print(f"[{epoch}/{args.epochs}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            (out_dir / "per_species_accuracy.json").write_text(json.dumps(per_species, indent=2))
            (out_dir / "top_confused_pairs.json").write_text(json.dumps(
                [{"true": a, "predicted_as": b, "count": c} for a, b, c in confused], indent=2))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Saved: best_model.pth, per_species_accuracy.json, top_confused_pairs.json")
    print("\nWorst-performing species (report these honestly in your README):")
    worst = sorted(
        ((k, v) for k, v in json.loads((out_dir / "per_species_accuracy.json").read_text()).items() if v is not None),
        key=lambda x: x[1]
    )[:5]
    for name, acc in worst:
        print(f"  {name}: {acc:.2f}")


if __name__ == "__main__":
    main()