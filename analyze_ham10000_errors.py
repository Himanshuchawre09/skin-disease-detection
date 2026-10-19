"""Evaluate and visualize errors of the trained HAM10000 ResNet50 checkpoint."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_CODES = ["nv", "mel", "bkl"]
CLASS_NAMES = ["Melanocytic nevus", "Melanoma", "Benign keratosis-like lesions"]
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class HAMTestDataset(Dataset):
    def __init__(self, dataframe, data_root, transform):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = self.data_root / str(row["path"])
        with Image.open(path) as source:
            image = source.convert("RGB")
        label = CLASS_CODES.index(str(row["dx"]))
        return self.transform(image), label, str(path), str(row["image_id"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    probabilities_all = []
    paths = []
    image_ids = []
    for images, labels, batch_paths, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predictions.tolist())
        probabilities_all.extend(probabilities.tolist())
        paths.extend(batch_paths)
        image_ids.extend(batch_ids)
    return (
        np.asarray(true_labels),
        np.asarray(predicted_labels),
        np.asarray(probabilities_all),
        paths,
        image_ids,
    )


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._capture_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, module, inputs, output):
        self.activations = output

    def _capture_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, tensor, target_index):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[0, target_index].backward()
        channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * self.activations).sum(dim=1)).squeeze()
        cam = cam.detach().float().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        self.model.zero_grad(set_to_none=True)
        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def choose_rows(errors, true_code, predicted_code, count):
    rows = errors[(errors["true_dx"] == true_code) & (errors["predicted_dx"] == predicted_code)]
    if len(rows) <= count:
        return rows
    return rows.sample(n=count, random_state=42).sort_values("image_id")


def save_error_grid(rows, output_path, title, columns=4):
    if rows.empty:
        return
    count = len(rows)
    row_count = int(np.ceil(count / columns))
    figure, axes = plt.subplots(row_count, columns, figsize=(4 * columns, 3.5 * row_count))
    axes = np.atleast_1d(axes).ravel()
    for axis, (_, item) in zip(axes, rows.iterrows()):
        with Image.open(item["path"]) as source:
            axis.imshow(source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        predicted_confidence = item[f"prob_{item['predicted_dx']}"]
        axis.set_title(
            f"True: {item['true_dx']} | Pred: {item['predicted_dx']}\n"
            f"Predicted-class confidence: {predicted_confidence:.1%}\n{item['image_id']}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[count:]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_melanoma_gradcam(rows, model, gradcam, transform, device, output_path):
    if rows.empty:
        return
    figure, axes = plt.subplots(len(rows), 3, figsize=(13, 4 * len(rows)), squeeze=False)
    for row_index, (_, item) in enumerate(rows.iterrows()):
        with Image.open(item["path"]) as source:
            image = source.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        predicted_index = CLASS_CODES.index(item["predicted_dx"])
        melanoma_index = CLASS_CODES.index("mel")
        predicted_cam = gradcam.generate(tensor, predicted_index)
        melanoma_cam = gradcam.generate(tensor, melanoma_index)

        image_array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[row_index, 0].imshow(image_array)
        axes[row_index, 0].set_title(
            f"True: melanoma | Predicted: {item['predicted_dx']}\n{item['image_id']}"
        )
        axes[row_index, 1].imshow(image_array)
        axes[row_index, 1].imshow(predicted_cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"Grad-CAM for wrong class: {item['predicted_dx']}")
        axes[row_index, 2].imshow(image_array)
        axes[row_index, 2].imshow(melanoma_cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_index, 2].set_title("Grad-CAM for melanoma score")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle("Melanoma False Negatives: Target-Class Grad-CAMs")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"
    manifest_path = data_root / "ham10000_3class" / "split_manifest.csv"
    checkpoint_path = base_dir / "runs" / "ham10000_resnet50_3class" / "best_model.pt"
    output_dir = base_dir / "runs" / "ham10000_resnet50_3class" / "error_analysis"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("class_names") != CLASS_CODES:
        raise ValueError(f"Unexpected checkpoint class order: {checkpoint.get('class_names')}")

    test_df = pd.read_csv(manifest_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = HAMTestDataset(test_df, data_root, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_CODES))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    true_labels, predicted_labels, probabilities, paths, image_ids = predict(model, loader, device)
    error_rows = []
    for index, (true_index, predicted_index) in enumerate(zip(true_labels, predicted_labels)):
        if true_index == predicted_index:
            continue
        record = {
            "image_id": image_ids[index],
            "path": paths[index],
            "true_dx": CLASS_CODES[true_index],
            "predicted_dx": CLASS_CODES[predicted_index],
        }
        record.update({f"prob_{code}": float(probabilities[index, j]) for j, code in enumerate(CLASS_CODES)})
        error_rows.append(record)
    errors = pd.DataFrame(error_rows)

    matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(CLASS_CODES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=range(len(CLASS_CODES)), zero_division=0
    )
    per_class = {
        code: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, code in enumerate(CLASS_CODES)
    }

    melanoma_fn = errors[errors["true_dx"] == "mel"]
    melanoma_fn_routes = {
        code: int((melanoma_fn["predicted_dx"] == code).sum()) for code in CLASS_CODES if code != "mel"
    }
    melanoma_fn_grid = pd.concat(
        [choose_rows(errors, "mel", "nv", 12), choose_rows(errors, "mel", "bkl", 12)],
        ignore_index=True,
    ).drop_duplicates("image_id")
    other_errors = errors[errors["true_dx"] != "mel"]
    other_grid_parts = []
    for (true_code, predicted_code), _ in other_errors.groupby(["true_dx", "predicted_dx"]):
        other_grid_parts.append(choose_rows(errors, true_code, predicted_code, 4))
    other_grid = (
        pd.concat(other_grid_parts, ignore_index=True).drop_duplicates("image_id")
        if other_grid_parts
        else pd.DataFrame()
    )
    gradcam_rows = pd.concat(
        [choose_rows(errors, "mel", "nv", 6), choose_rows(errors, "mel", "bkl", 6)],
        ignore_index=True,
    ).drop_duplicates("image_id")

    output_dir.mkdir(parents=True, exist_ok=False)
    errors.to_csv(output_dir / "misclassified_predictions.csv", index=False)
    save_error_grid(
        melanoma_fn_grid,
        output_dir / "melanoma_false_negatives.png",
        "Representative Melanoma False Negatives",
    )
    save_error_grid(
        other_grid,
        output_dir / "other_class_confusions.png",
        "Representative Non-melanoma Class Confusions",
    )
    gradcam = GradCAM(model, model.layer4[-1].conv3)
    save_melanoma_gradcam(
        gradcam_rows,
        model,
        gradcam,
        transform,
        device,
        output_dir / "melanoma_false_negative_gradcam.png",
    )
    gradcam.close()

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_CODES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "test_manifest": str(manifest_path),
        "test_images": int(len(test_df)),
        "errors": int(len(errors)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_codes": CLASS_CODES,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "per_class": per_class,
        "melanoma_false_negatives": int(len(melanoma_fn)),
        "melanoma_false_negative_routes": melanoma_fn_routes,
        "other_error_confusions": {
            f"{true_code}_to_{predicted_code}": int(count)
            for (true_code, predicted_code), count in other_errors.groupby(
                ["true_dx", "predicted_dx"]
            ).size().items()
        },
        "visual_samples": {
            "melanoma_false_negative_grid": int(len(melanoma_fn_grid)),
            "other_confusion_grid": int(len(other_grid)),
            "gradcam_melanoma_false_negatives": int(len(gradcam_rows)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"Device: {device}")
    print(f"Test images: {len(test_df)} | misclassified: {len(errors)}")
    print(f"Melanoma false negatives: {len(melanoma_fn)}")
    print(f"Melanoma false-negative routes: {melanoma_fn_routes}")
    print("Other confusion pairs:")
    for pair, count in summary["other_error_confusions"].items():
        print(f"  {pair}: {count}")
    print(f"Saved analysis artifacts: {output_dir}")


if __name__ == "__main__":
    main()