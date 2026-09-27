import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EczemaSeverityDataset(Dataset):
    def __init__(self, dataframe, image_root, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def _resolve_path(self, row):
        path = self.image_root / str(row["path"])
        if path.exists():
            return path

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

        tensor = self.transform(image)
        label = SEVERITY_CLASSES.index(str(row["severity"]))
        return tensor, label, str(path)


def build_model(checkpoint, device):
    arch = checkpoint.get("arch", checkpoint.get("model_name", "resnet50")).lower()
    if "resnet" in arch:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(SEVERITY_CLASSES))
    elif "efficientnet" in arch:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(SEVERITY_CLASSES))
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), arch


def evaluate(model, loader, device):
    true_labels = []
    predicted_labels = []
    confidences = []
    paths = []

    with torch.no_grad():
        for tensors, labels, batch_paths in loader:
            logits = model(tensors.to(device))
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().numpy()

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(predictions.tolist())
            confidences.extend(probabilities.max(dim=1).values.cpu().numpy().tolist())
            paths.extend(batch_paths)

    return np.asarray(true_labels), np.asarray(predicted_labels), np.asarray(confidences), paths


def save_confusion_matrix(true_labels, predicted_labels, output_path):
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SEVERITY_CLASSES,
        yticklabels=SEVERITY_CLASSES,
    )
    plt.title("Eczema Severity Confusion Matrix")
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def save_misclassification_grid(
    true_labels, predicted_labels, confidences, paths, output_path, max_examples
):
    indices = np.flatnonzero(true_labels != predicted_labels)[:max_examples]
    if len(indices) == 0:
        return 0

    columns = 4
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, indices):
        axis.imshow(Image.open(paths[index]).convert("RGB"))
        axis.set_title(
            f"True: {SEVERITY_CLASSES[true_labels[index]]}\n"
            f"Pred: {SEVERITY_CLASSES[predicted_labels[index]]} "
            f"({confidences[index] * 100:.1f}%)\n{Path(paths[index]).name[:32]}",
            color="crimson",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(indices) :]:
        axis.axis("off")

    figure.suptitle("Eczema Severity Misclassified Test Examples", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description="Diagnose the Eczema severity checkpoint")
    parser.add_argument("--manifest", default="data/dataset_manifest.csv")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--checkpoint", default="runs/severity_eczema/best_model.pt")
    parser.add_argument("--output-dir", default="runs/severity_eczema/diagnosis")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    manifest_path = (base_dir / args.manifest).resolve()
    image_root = (base_dir / args.image_root).resolve()
    checkpoint_path = (base_dir / args.checkpoint).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    print(f"Using device: {device}")

    manifest = pd.read_csv(manifest_path)
    eczema_test = manifest[
        (manifest["disease"] == "Eczema") & (manifest["split"] == "test")
    ].copy()
    if eczema_test.empty:
        raise ValueError("No Eczema test rows found in the manifest")

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = EczemaSeverityDataset(eczema_test, image_root, eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, arch = build_model(checkpoint, device)
    true_labels, predicted_labels, confidences, paths = evaluate(model, loader, device)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        zero_division=0,
    )
    accuracy = float((true_labels == predicted_labels).mean())
    mae = float(mean_absolute_error(true_labels, predicted_labels))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=SEVERITY_CLASSES,
        digits=4,
        zero_division=0,
    )

    save_confusion_matrix(
        true_labels, predicted_labels, output_dir / "confusion_matrix.png"
    )
    misclassified_count = save_misclassification_grid(
        true_labels,
        predicted_labels,
        confidences,
        paths,
        output_dir / "misclassified_grid.png",
        args.max_examples,
    )

    metrics = {
        "disease": "Eczema",
        "arch": arch,
        "test_examples": int(len(true_labels)),
        "misclassified_examples": int((true_labels != predicted_labels).sum()),
        "test_accuracy": accuracy,
        "ordinal_mae": mae,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(SEVERITY_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=[0, 1, 2]
        ).tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)
        file.write(f"\nAccuracy: {accuracy:.6f}\nOrdinal MAE: {mae:.6f}\n")

    predictions = eczema_test[["filename", "category", "severity"]].copy()
    predictions["true_index"] = true_labels
    predictions["predicted_severity"] = [SEVERITY_CLASSES[index] for index in predicted_labels]
    predictions["predicted_index"] = predicted_labels
    predictions["confidence"] = confidences
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    print(f"Eczema test examples: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Ordinal MAE: {mae:.4f}")
    print(report)
    print(f"Saved diagnosis artifacts to: {output_dir}")
    print(f"Misclassification grid examples saved: {misclassified_count}")


if __name__ == "__main__":
    main()