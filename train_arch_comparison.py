import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mean_absolute_error, precision_recall_fscore_support
from torch.utils.data import DataLoader

from arch_variants import build_model
from train_severity_ordinal_pytorch import (
    SEVERITY_CLASSES,
    SeverityDataset,
    build_transforms,
    predict_ordinal,
    save_confusion,
    seed_everything,
    summarize,
)


def parse_networks(value):
    if value is None or value == "all":
        return ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "squeezenet1_1", "shufflenet_v2_x1_0", "googlenet"]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_variants(value):
    if value is None or value == "all":
        return ["baseline", "remove", "add"]
    return [item.strip() for item in value.split(",") if item.strip()]


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_inference_ms(model, loader, device):
    model.eval()
    timings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            start = time.perf_counter()
            _ = model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms / max(1, images.size(0)))
    if not timings:
        return 0.0
    return float(np.mean(timings))


def build_dataloaders(manifest, image_root, disease, batch_size, num_workers, train_transform, eval_transform, device):
    disease_df = manifest[(manifest["disease"] == disease) & manifest["severity"].isin(SEVERITY_CLASSES)].copy()
    train_df = disease_df[disease_df["split"] == "train"].copy()
    val_df = disease_df[disease_df["split"] == "val"].copy()
    test_df = disease_df[disease_df["split"] == "test"].copy()
    loaders = {
        "train": DataLoader(
            SeverityDataset(train_df, image_root, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "val": DataLoader(
            SeverityDataset(val_df, image_root, eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "test": DataLoader(
            SeverityDataset(test_df, image_root, eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        ),
    }
    return train_df, val_df, test_df, loaders


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    from train_severity_ordinal_pytorch import run_epoch as existing_run_epoch

    return existing_run_epoch(model, loader, criterion, optimizer, scaler, device, train)


def save_predictions_csv(output_path, true_values, predicted_values, raw_values, rows):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", "filename", "true_label", "predicted_label", "raw_value", "path"])
        for index, row in enumerate(rows):
            writer.writerow([
                index,
                row["filename"],
                int(true_values[index]),
                int(predicted_values[index]),
                float(raw_values[index]),
                str(row["path"]),
            ])


def evaluate_run(model, test_loader, device):
    true_values, predicted_values, raw_values = predict_ordinal(model, test_loader, device)
    pred_clip = np.clip(np.rint(raw_values), 0, 2).astype(int)
    summary = summarize(true_values, pred_clip)
    return true_values, pred_clip, raw_values, summary


def train_one_run(args, disease, network, variant, manifest, image_root, output_root):
    run_dir = output_root / disease / f"{network}_{variant}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"SKIP {disease}/{network}/{variant}: metrics exists -> {metrics_path}")
        return {"status": "skipped", "path": str(metrics_path)}

    run_dir.mkdir(parents=True, exist_ok=True)
    error_log = run_dir / "error.txt"
    status = {
        "training": False,
        "validation": False,
        "checkpoint": False,
        "metrics": False,
        "predictions": False,
    }
    try:
        train_transform, eval_transform = build_transforms()
        train_df, val_df, test_df, loaders = build_dataloaders(
            manifest,
            image_root,
            disease,
            args.batch_size,
            args.num_workers,
            train_transform,
            eval_transform,
            get_device(args.device),
        )
        model, feature_info, description = build_model(network, variant)
        model = model.to(get_device(args.device))
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)
        scaler = torch.amp.GradScaler("cuda", enabled=(get_device(args.device).type == "cuda"))

        best_state = None
        best_val_mae = math.inf
        best_epoch = 0
        no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_mae": [], "seconds_per_epoch": []}

        device = get_device(args.device)
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_mae = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, True)
            val_loss, val_mae = run_epoch(model, loaders["val"], criterion, optimizer, scaler, device, False)
            scheduler.step(val_mae)
            elapsed = time.perf_counter() - epoch_start
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_mae"].append(float(val_mae))
            history["seconds_per_epoch"].append(float(elapsed))
            print(f"{disease}/{network}/{variant} epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f} seconds={elapsed:.2f}")
            if val_mae < best_val_mae - 1e-4:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                print(f"EARLY STOPPING {disease}/{network}/{variant} after epoch {epoch}")
                break

        if best_state is None:
            raise RuntimeError("No valid checkpoint was created during training")

        torch.save(
            {
                "disease": disease,
                "network": network,
                "variant": variant,
                "best_epoch": best_epoch,
                "best_val_mae": float(best_val_mae),
                "state_dict": best_state,
            },
            run_dir / "best_model.pt",
        )
        status["checkpoint"] = True
        model.load_state_dict(best_state)
        model.to(device)
        status["training"] = True
        status["validation"] = True

        with (run_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        true_values, predicted_values, raw_values, summary = evaluate_run(model, loaders["test"], device)
        average_epoch_seconds = float(np.mean(history["seconds_per_epoch"])) if history["seconds_per_epoch"] else 0.0
        inference_ms = compute_inference_ms(model, loaders["test"], device)

        conf_matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1, 2]).tolist()
        precision, recall, f1, support = precision_recall_fscore_support(
            true_values,
            predicted_values,
            labels=[0, 1, 2],
            zero_division=0,
        )
        per_class = {
            SEVERITY_CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(SEVERITY_CLASSES))
        }

        metrics = {
            "disease": disease,
            "network": network,
            "variant": variant,
            "best_epoch": int(best_epoch),
            "best_val_mae": float(best_val_mae),
            "test_accuracy": float(summary["accuracy"]),
            "test_mae": float(summary["ordinal_mae"]),
            "macro_precision": float(summary["macro_precision"]),
            "macro_recall": float(summary["macro_recall"]),
            "macro_f1": float(summary["macro_f1"]),
            "per_class": per_class,
            "confusion_matrix": conf_matrix,
            "param_count_total": int(total_params),
            "param_count_trainable": int(trainable_params),
            "average_training_seconds_per_epoch": float(average_epoch_seconds),
            "inference_ms_per_image": float(inference_ms),
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        status["metrics"] = True

        test_rows = manifest[(manifest["disease"] == disease) & (manifest["split"] == "test")].copy()
        save_predictions_csv(run_dir / "predictions.csv", true_values, predicted_values, raw_values, test_rows.to_dict("records"))
        status["predictions"] = True

        save_confusion(true_values, predicted_values, run_dir / "confusion_matrix.png", f"{disease} {network}/{variant}")

        print(
            f"SUMMARY {disease}/{network}/{variant}: "
            f"best_epoch={best_epoch} best_val_mae={best_val_mae:.4f} "
            f"test_accuracy={summary['accuracy']:.4f} test_mae={summary['ordinal_mae']:.4f} "
            f"inference_ms={inference_ms:.3f}"
        )
        return {
            "status": "ok",
            "path": str(run_dir),
            "metrics": metrics,
            "status_flags": status,
        }
    except Exception as exc:
        with error_log.open("w", encoding="utf-8") as file:
            file.write(f"{type(exc).__name__}: {exc}\n")
        print(f"ERROR {disease}/{network}/{variant}: {type(exc).__name__}: {exc}")
        return {"status": "error", "path": str(run_dir), "error": str(exc), "status_flags": status}


def main():
    parser = argparse.ArgumentParser(description="Architecture comparison runner for ordinal severity models")
    parser.add_argument("--disease", choices=["Eczema", "Acne"], default="Eczema")
    parser.add_argument("--networks", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--manifest", default="data/dataset_manifest_v3.csv")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--output-root", default="runs/arch_comparison")
    args = parser.parse_args()

    seed_everything(args.seed)
    base_dir = Path(__file__).resolve().parent
    manifest_path = (base_dir / args.manifest).resolve()
    image_root = (base_dir / args.image_root).resolve()
    output_root = (base_dir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    networks = parse_networks(args.networks)
    variants = parse_variants(args.variants)

    results = []
    for network in networks:
        for variant in variants:
            result = train_one_run(args, args.disease, network, variant, manifest, image_root, output_root)
            results.append(result)

    return results


if __name__ == "__main__":
    main()
