"""Train a ResNet50 baseline on the lesion-grouped HAM10000 nv/mel/bkl split."""

import argparse
import copy
import json
import math
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
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


SEED = 42
CLASS_CODES = ["nv", "mel", "bkl"]
CLASS_NAMES = ["Melanocytic nevus", "Melanoma", "Benign keratosis-like lesions"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, data_root, transform):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(CLASS_CODES)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.data_root / str(row["path"])
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read HAM10000 image: {image_path}") from exc
        return self.transform(image), self.class_to_idx[str(row["dx"])]


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


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train(train)
    total_loss = 0.0
    labels_all = []
    predictions_all = []
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
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return (
        total_loss / len(loader.dataset),
        float(accuracy_score(labels_all, predictions_all)),
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels_all = []
    predictions_all = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
        labels_all.extend(labels.numpy().tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(labels_all), np.asarray(predictions_all)


def save_confusion_matrix(matrix, path, model_name):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_CODES,
        yticklabels=CLASS_CODES,
    )
    plt.title(f"HAM10000 {model_name} Test Confusion Matrix")
    plt.xlabel("Predicted dx")
    plt.ylabel("True dx")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ham10000_3class/split_manifest.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/ham10000_resnet50_3class")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything()
    base_dir = Path(__file__).resolve().parent
    manifest_path = (base_dir / args.manifest).resolve()
    data_root = (base_dir / args.data_root).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing HAM10000 run: {output_dir}")

    manifest = pd.read_csv(manifest_path)
    required_columns = {"dx", "lesion_id", "path", "split"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Prepared split manifest missing columns: {sorted(missing)}")
    if set(manifest["dx"].unique()) != set(CLASS_CODES):
        raise ValueError(f"Expected only dx classes {CLASS_CODES}")
    if manifest["path"].map(lambda item: (data_root / str(item)).is_file()).sum() != len(manifest):
        raise FileNotFoundError("One or more image paths in the prepared manifest do not exist")
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        first_groups = set(manifest.loc[manifest["split"] == first, "lesion_id"])
        second_groups = set(manifest.loc[manifest["split"] == second, "lesion_id"])
        if first_groups & second_groups:
            raise ValueError(f"lesion_id overlap found between {first} and {second}")

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()
    print(f"Device: {'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"Images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Training class counts:")
    print(train_df["dx"].value_counts().reindex(CLASS_CODES).to_string())

    train_transform, eval_transform = build_transforms()
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        HAM10000Dataset(train_df, data_root, train_transform),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_df, data_root, eval_transform),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        HAM10000Dataset(test_df, data_root, eval_transform),
        shuffle=False,
        **loader_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_CODES))
    else:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_CODES))
    model.to(device)

    class_counts = train_df["dx"].value_counts().reindex(CLASS_CODES)
    class_weights = len(train_df) / (len(CLASS_CODES) * class_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} | val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_name": args.model,
                    "class_names": CLASS_CODES,
                    "class_display_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_class_counts": class_counts.astype(int).to_dict(),
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "split_manifest": str(manifest_path),
                    "split_group_column": "lesion_id",
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training ended without a best validation checkpoint")

    model.load_state_dict(best_state)
    y_true, y_pred = predict(model, test_loader, device)
    labels = list(range(len(CLASS_CODES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    metrics = {
        "model": args.model,
        "classes": CLASS_CODES,
        "test_images": int(len(y_true)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": accuracy,
        "test_macro_precision": float(precision.mean()),
        "test_macro_recall": float(recall.mean()),
        "test_macro_f1": float(f1.mean()),
        "per_class": {
            code: {
                "display_name": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, code in enumerate(CLASS_CODES)
        },
        "confusion_matrix": {
            "labels": CLASS_CODES,
            "rows_true_columns_predicted": matrix.tolist(),
        },
    }

    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png", args.model)
    save_history(history, output_dir / "training_curves.png")

    print("\nTEST RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision.mean():.4f}")
    print(f"Macro recall: {recall.mean():.4f}")
    print(f"Macro F1: {f1.mean():.4f}")
    print("\nPer-class metrics:")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted; labels nv, mel, bkl):")
    print(matrix)
    print(f"\nArtifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()