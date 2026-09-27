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
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

SEED = 42
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
IMAGE_SIZE = 224


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SkinSeverityDataset(Dataset):
    def __init__(self, manifest_df, image_root, transform=None):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(SEVERITY_CLASSES)}

    def _resolve_path(self, row):
        filename = str(row["filename"])
        rel_path = str(row.get("path", ""))

        # 1. Try manifest direct relative path
        if rel_path:
            p0 = self.image_root / rel_path
            if p0.exists():
                return p0

        # 2. Try clean_raw layout (category/filename)
        category = str(row.get("category", ""))
        if category:
            p1 = self.image_root / "clean_raw" / category / filename
            if p1.exists():
                return p1

        # 3. Recursive lookup
        matches = list(self.image_root.rglob(filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Image not found: {filename} under {self.image_root}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row)

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc

        label = self.class_to_idx[str(row["severity"])]

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


def make_severity_model(arch: str = "resnet50"):
    arch = arch.lower()
    if arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(SEVERITY_CLASSES))
        return model
    elif arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(SEVERITY_CLASSES))
        return model
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
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
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SEVERITY_CLASSES))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SEVERITY_CLASSES,
        yticklabels=SEVERITY_CLASSES,
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_training_plot(history, out_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="navy")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="crimson", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="navy")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="crimson", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def train_single_severity_model(
    disease: str,
    manifest_df: pd.DataFrame,
    image_root: str,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    patience: int,
    out_dir: Path,
    device: torch.device,
):
    print("\n" + "=" * 70)
    print(f"TRAINING STAGE 2 SEVERITY MODEL FOR: {disease.upper()}")
    print("=" * 70)

    # Filter data for this specific disease
    disease_df = manifest_df[manifest_df["disease"] == disease].copy()
    disease_df = disease_df[disease_df["severity"].isin(SEVERITY_CLASSES)].copy()

    train_df = disease_df[disease_df["split"] == "train"].copy()
    val_df = disease_df[disease_df["split"] == "val"].copy()
    test_df = disease_df[disease_df["split"] == "test"].copy()

    print(f"Dataset sizes for {disease} Severity:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nClass distribution in Training Split:")
    print(train_df["severity"].value_counts().reindex(SEVERITY_CLASSES))

    train_tfms, eval_tfms = build_transforms()

    train_ds = SkinSeverityDataset(train_df, image_root, train_tfms)
    val_ds = SkinSeverityDataset(val_df, image_root, eval_tfms)
    test_ds = SkinSeverityDataset(test_df, image_root, eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = make_severity_model(arch).to(device)

    # Balanced class weights
    counts = train_df["severity"].value_counts().reindex(SEVERITY_CLASSES).values
    weights = len(train_df) / (len(SEVERITY_CLASSES) * counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model_dir = out_dir / f"severity_{disease.lower()}"
    model_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\nStarting {disease} Severity Training ({epochs} epochs max)...")

    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch:02d}/{epochs} | "
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
                    "disease": disease,
                    "arch": arch,
                    "severity_classes": SEVERITY_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                model_dir / "best_model.pt",
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved.")

    model.load_state_dict(best_state)

    # Test set evaluation
    y_true, y_pred = predict(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(SEVERITY_CLASSES))), average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(SEVERITY_CLASSES))),
        target_names=SEVERITY_CLASSES,
        digits=4,
        zero_division=0,
    )

    print(f"\n================ {disease.upper()} SEVERITY TEST RESULTS ================")
    print(f"Test Accuracy  : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print("\nPer-Class Report:")
    print(report)

    # Save artifacts
    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nOverall accuracy: {accuracy:.6f}\n")
        f.write(f"Macro precision: {precision:.6f}\n")
        f.write(f"Macro recall: {recall:.6f}\n")
        f.write(f"Macro F1: {f1:.6f}\n")

    metrics = {
        "disease": disease,
        "arch": arch,
        "test_accuracy": float(accuracy),
        "test_macro_precision": float(precision),
        "test_macro_recall": float(recall),
        "test_macro_f1": float(f1),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(model_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_confusion_matrix(
        y_true,
        y_pred,
        model_dir / "confusion_matrix.png",
        f"{disease} Severity Confusion Matrix",
    )
    save_training_plot(
        history,
        model_dir / "training_curves.png",
        f"{disease} Severity",
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Severity Assessment Models")
    parser.add_argument("--manifest", default=r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data\dataset_manifest.csv")
    parser.add_argument("--image-root", default=r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data")
    parser.add_argument("--disease", choices=["all", "Acne", "Eczema"], default="all")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default=r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\runs")
    args = parser.parse_args()

    seed_everything()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. Running on CPU.")

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diseases = ["Acne", "Eczema"] if args.disease == "all" else [args.disease]

    all_metrics = {}
    for dis in diseases:
        metrics = train_single_severity_model(
            disease=dis,
            manifest_df=manifest,
            image_root=args.image_root,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            patience=args.patience,
            out_dir=out_dir,
            device=device,
        )
        all_metrics[dis] = metrics

    print("\n" + "=" * 70)
    print("STAGE 2 SEVERITY TRAINING COMPLETED FOR ALL TARGET DISEASES!")
    print("=" * 70)
    for dis, met in all_metrics.items():
        print(f"  {dis:10s} -> Test Accuracy: {met['test_accuracy']*100:.2f}% | Macro F1: {met['test_macro_f1']*100:.2f}% (Best Epoch: {met['best_epoch']})")


if __name__ == "__main__":
    main()
