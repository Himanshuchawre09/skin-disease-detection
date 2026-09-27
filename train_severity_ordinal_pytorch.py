import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 42
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SeverityDataset(Dataset):
    def __init__(self, dataframe, image_root, transform):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def _resolve_path(self, row):
        direct = self.image_root / str(row["path"])
        if direct.exists():
            return direct
        fallback = self.image_root / "clean_raw" / str(row["category"]) / str(row["filename"])
        if fallback.exists():
            return fallback
        matches = list(self.image_root.rglob(str(row["filename"])))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Image not found: {row['filename']}")

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = self._resolve_path(row)
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc
        return self.transform(image), float(SEVERITY_CLASSES.index(str(row["severity"])))


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.00), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform


def make_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train(train)
    total_loss = 0.0
    true_values = []
    predicted_values = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        true_values.extend(labels.detach().cpu().numpy().ravel().tolist())
        predicted_values.extend(outputs.detach().cpu().numpy().ravel().tolist())

    return (
        total_loss / len(loader.dataset),
        mean_absolute_error(true_values, predicted_values),
    )


@torch.no_grad()
def predict_ordinal(model, loader, device):
    model.eval()
    true_values = []
    raw_values = []
    for images, labels in loader:
        outputs = model(images.to(device, non_blocking=True)).squeeze(1)
        true_values.extend(labels.numpy().tolist())
        raw_values.extend(outputs.cpu().numpy().tolist())
    true_values = np.asarray(true_values, dtype=int)
    raw_values = np.asarray(raw_values)
    predicted_values = np.clip(np.rint(raw_values), 0, 2).astype(int)
    return true_values, predicted_values, raw_values


@torch.no_grad()
def predict_categorical(model, loader, device):
    model.eval()
    true_values = []
    predicted_values = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        true_values.extend(labels.numpy().tolist())
        predicted_values.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(true_values, dtype=int), np.asarray(predicted_values, dtype=int)


def load_categorical_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", checkpoint.get("model_name", "resnet50")).lower()
    if "resnet" not in arch:
        raise ValueError(f"Expected a ResNet categorical checkpoint, got {arch}")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SEVERITY_CLASSES))
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    return model.to(device).eval()


def save_confusion(true_values, predicted_values, output_path, title):
    matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES)
    plt.title(title)
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def summarize(true_values, predicted_values):
    precision, recall, f1, support = precision_recall_fscore_support(
        true_values, predicted_values, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "ordinal_mae": float(mean_absolute_error(true_values, predicted_values)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(SEVERITY_CLASSES)
        },
        "confusion_matrix": confusion_matrix(true_values, predicted_values, labels=[0, 1, 2]).tolist(),
    }


def train_disease(
    disease,
    manifest,
    image_root,
    output_root,
    device,
    epochs,
    batch_size,
    lr,
    patience,
    output_dir=None,
    resume_checkpoint=None,
):
    disease_df = manifest[
        (manifest["disease"] == disease) & manifest["severity"].isin(SEVERITY_CLASSES)
    ].copy()
    train_df = disease_df[disease_df["split"] == "train"]
    val_df = disease_df[disease_df["split"] == "val"]
    test_df = disease_df[disease_df["split"] == "test"]
    train_transform, eval_transform = build_transforms()
    loaders = {
        "train": DataLoader(SeverityDataset(train_df, image_root, train_transform), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")),
        "val": DataLoader(SeverityDataset(val_df, image_root, eval_transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
        "test": DataLoader(SeverityDataset(test_df, image_root, eval_transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
    }
    output_dir = output_dir or output_root / f"severity_{disease.lower()}_ordinal"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_state = None
    best_val_mae = math.inf
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": [], "lr": []}
    start_epoch = 1

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        best_val_mae = float(checkpoint.get("best_val_mae", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = best_epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "disease": disease,
                "arch": "resnet50",
                "ordinal": True,
                "state_dict": best_state,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
            },
            output_dir / "best_model.pt",
        )
        print(f"Resuming {disease.upper()} from epoch {best_epoch}; training through epoch {epochs}.")

    print(f"\nTRAINING ORDINAL {disease.upper()} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_mae = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, True)
        val_loss, val_mae = run_epoch(model, loaders["val"], criterion, optimizer, scaler, device, False)
        scheduler.step(val_mae)
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch:02d}/{epochs} | train_mae={train_mae:.4f} | val_mae={val_mae:.4f} | val_loss={val_loss:.4f}")
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"disease": disease, "arch": "resnet50", "ordinal": True, "state_dict": best_state, "best_val_mae": best_val_mae, "best_epoch": best_epoch}, output_dir / "best_model.pt")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    ordinal_true, ordinal_pred, raw_values = predict_ordinal(model, loaders["test"], device)
    ordinal_summary = summarize(ordinal_true, ordinal_pred)
    save_confusion(ordinal_true, ordinal_pred, output_dir / "confusion_matrix.png", f"Ordinal {disease} Severity")
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(classification_report(ordinal_true, ordinal_pred, labels=[0, 1, 2], target_names=SEVERITY_CLASSES, digits=4, zero_division=0))
        file.write(f"\nAccuracy: {ordinal_summary['accuracy']:.6f}\nOrdinal MAE: {ordinal_summary['ordinal_mae']:.6f}\n")
    with open(output_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"disease": disease, "best_epoch": best_epoch, "best_val_mae": best_val_mae, **ordinal_summary}, file, indent=2)
    return ordinal_summary, loaders["test"], ordinal_true


def main():
    parser = argparse.ArgumentParser(description="Train ordinal severity models without changing dataset splits")
    parser.add_argument("--manifest", default="data/dataset_manifest.csv")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disease", choices=["all", "Acne", "Eczema"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    seed_everything()
    base_dir = Path(__file__).resolve().parent
    manifest = pd.read_csv(base_dir / args.manifest)
    image_root = (base_dir / args.image_root).resolve()
    output_root = (base_dir / args.output_root).resolve()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"Using device: {device}")
    comparison = {}
    diseases = ["Acne", "Eczema"] if args.disease == "all" else [args.disease]
    for disease in diseases:
        ordinal_summary, test_loader, true_values = train_disease(
            disease,
            manifest,
            image_root,
            output_root,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
            output_dir=(base_dir / args.output_dir).resolve() if args.output_dir else None,
            resume_checkpoint=(base_dir / args.resume).resolve() if args.resume else None,
        )
        categorical_model = load_categorical_model(base_dir / "runs" / f"severity_{disease.lower()}" / "best_model.pt", device)
        categorical_true, categorical_pred = predict_categorical(categorical_model, test_loader, device)
        categorical_summary = summarize(categorical_true, categorical_pred)
        comparison[disease] = {"categorical_original": categorical_summary, "ordinal": ordinal_summary}
        print(f"{disease}: categorical accuracy={categorical_summary['accuracy']:.4f}, MAE={categorical_summary['ordinal_mae']:.4f} | ordinal accuracy={ordinal_summary['accuracy']:.4f}, MAE={ordinal_summary['ordinal_mae']:.4f}")
    with open(output_root / "severity_ordinal_comparison.json", "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
    print(f"Saved comparison to: {output_root / 'severity_ordinal_comparison.json'}")


if __name__ == "__main__":
    main()