
import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

SEED = 42
CLASS_NAMES = ["Acne", "Eczema", "Normal"]
IMAGE_SIZE = 224


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SkinDataset(Dataset):
    def __init__(self, manifest_df, image_root, transform=None):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    def _resolve_path(self, row):
        filename = str(row["filename"])
        disease = str(row["disease"])

        # First try disease/filename (expected clean_raw layout).
        p1 = self.image_root / disease / filename
        if p1.exists():
            return p1

        # Then try filename directly under image_root.
        p2 = self.image_root / filename
        if p2.exists():
            return p2

        # Finally search recursively, which supports a slightly different folder layout.
        matches = list(self.image_root.rglob(filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Image not found: {filename}. "
            f"Checked {p1}, {p2}, and recursive search below {self.image_root}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row)

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc

        label = self.class_to_idx[str(row["disease"])]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.80, 1.00), ratio=(0.90, 1.10)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tfms, eval_tfms


def make_model(name: str):
    name = name.lower()

    if name in {"efficientnet_b0", "efficientnet"}:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        return model

    if name in {"resnet50", "resnet"}:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(CLASS_NAMES))
        return model

    raise ValueError("Model must be 'efficientnet_b0' or 'resnet50'.")


def make_class_weights(train_df):
    counts = train_df["disease"].value_counts()
    counts = counts.reindex(CLASS_NAMES).fillna(0)
    if (counts <= 0).any():
        raise ValueError(f"Missing classes in training split: {counts.to_dict()}")

    n = counts.sum()
    c = len(CLASS_NAMES)
    weights = n / (c * counts)
    weights = weights / weights.mean()

    return torch.tensor(weights.values, dtype=torch.float32), counts.astype(int).to_dict()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)

        preds = logits.argmax(dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_training_plot(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model", choices=["efficientnet_b0", "resnet50"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    seed_everything()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. Training will be much slower.")

    manifest = pd.read_csv(args.manifest)

    required_cols = {"filename", "disease", "split"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    manifest = manifest[manifest["disease"].isin(CLASS_NAMES)].copy()

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/val/test split is missing one or more partitions.")

    print("\nDataset sizes:")
    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))

    print("\nTraining class counts:")
    print(train_df["disease"].value_counts().reindex(CLASS_NAMES))

    train_tfms, eval_tfms = build_transforms()

    train_ds = SkinDataset(train_df, args.image_root, train_tfms)
    val_ds = SkinDataset(val_df, args.image_root, eval_tfms)
    test_ds = SkinDataset(test_df, args.image_root, eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = make_model(args.model).to(device)

    class_weights, counts = make_class_weights(train_df)
    class_weights = class_weights.to(device)
    print("\nComputed class weights:", dict(zip(CLASS_NAMES, class_weights.detach().cpu().tolist())))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_dir = Path(args.out_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    print("\nStarting training...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True
        )

        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_name": args.model,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "class_counts": counts,
                    "class_weights": class_weights.detach().cpu().tolist(),
                },
                run_dir / "best_model.pt",
            )
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved.")

    model.load_state_dict(best_state)

    y_true, y_pred = predict(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n================ TEST RESULTS ================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print("\nPer-class report:")
    print(report)

    with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
        f.write(f"Overall accuracy: {accuracy:.6f}\n")
        f.write(f"Macro precision: {precision:.6f}\n")
        f.write(f"Macro recall: {recall:.6f}\n")
        f.write(f"Macro F1: {f1:.6f}\n")

    metrics = {
        "model": args.model,
        "test_accuracy": accuracy,
        "test_macro_precision": precision,
        "test_macro_recall": recall,
        "test_macro_f1": f1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_training_plot(history, run_dir / "training_curves.png")
    save_confusion_matrix(
        y_true,
        y_pred,
        run_dir / "confusion_matrix.png",
        f"{args.model}: Test Confusion Matrix",
    )

    print("\nSaved:")
    print(run_dir / "best_model.pt")
    print(run_dir / "classification_report.txt")
    print(run_dir / "metrics.json")
    print(run_dir / "training_curves.png")
    print(run_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
