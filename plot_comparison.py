from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("runs/arch_comparison")
NETWORK_COLORS = {
    "resnet50": "#1f77b4",
    "densenet121": "#ff7f0e",
    "efficientnet_b0": "#2ca02c",
    "mobilenet_v2": "#d62728",
    "squeezenet1_1": "#9467bd",
    "shufflenet_v2_x1_0": "#8c564b",
    "googlenet": "#17becf",
}
VARIANT_STYLES = {
    "baseline": "-",
    "remove": "--",
    "add": ":",
}


def discover_runs(disease_dir: Path):
    runs = []
    for run_dir in sorted(disease_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        history_file = run_dir / "history.json"
        metrics_file = run_dir / "metrics.json"
        if not history_file.exists() or not metrics_file.exists():
            continue
        with history_file.open("r", encoding="utf-8") as f:
            history = json.load(f)
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        runs.append({"dir": run_dir, "history": history, "metrics": metrics})
    return runs


def format_number(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 1_000:
            return f"{value:.3f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def build_master_rows(disease_dir: Path):
    rows = []
    for run in discover_runs(disease_dir):
        metrics = run["metrics"]
        history = run["history"]
        per_class = metrics.get("per_class", {})
        cm = metrics.get("confusion_matrix", [])
        network = metrics.get("network", run["dir"].name.split("_", 1)[0])
        variant = metrics.get("variant", run["dir"].name.rsplit("_", 1)[-1])
        row = {
            "disease": disease_dir.name,
            "network": network,
            "variant": variant,
            "best_epoch": metrics.get("best_epoch", history.get("best_epoch")),
            "best_val_mae": metrics.get("best_val_mae"),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_mae": metrics.get("test_mae"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "param_count_total": metrics.get("param_count_total"),
            "param_count_trainable": metrics.get("param_count_trainable"),
            "average_training_seconds_per_epoch": metrics.get("average_training_seconds_per_epoch"),
            "inference_ms_per_image": metrics.get("inference_ms_per_image"),
        }
        for label in ["Mild", "Moderate", "Severe"]:
            stats = per_class.get(label, {})
            row[f"{label}_precision"] = stats.get("precision")
            row[f"{label}_recall"] = stats.get("recall")
            row[f"{label}_f1"] = stats.get("f1")
            row[f"{label}_support"] = stats.get("support")
        for i in range(3):
            for j in range(3):
                val = cm[i][j] if len(cm) > i and len(cm[i]) > j else None
                row[f"confusion_{i}{j}"] = val
        rows.append(row)
    return sorted(rows, key=lambda r: (float(r.get("test_mae", 999.0)), r["network"], r["variant"]))


def write_master_table(rows, disease_dir: Path):
    output_dir = disease_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "disease",
        "network",
        "variant",
        "best_epoch",
        "best_val_mae",
        "test_accuracy",
        "test_mae",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "param_count_total",
        "param_count_trainable",
        "average_training_seconds_per_epoch",
        "inference_ms_per_image",
        "Mild_precision",
        "Mild_recall",
        "Mild_f1",
        "Mild_support",
        "Moderate_precision",
        "Moderate_recall",
        "Moderate_f1",
        "Moderate_support",
        "Severe_precision",
        "Severe_recall",
        "Severe_f1",
        "Severe_support",
        "confusion_00",
        "confusion_01",
        "confusion_02",
        "confusion_10",
        "confusion_11",
        "confusion_12",
        "confusion_20",
        "confusion_21",
        "confusion_22",
    ]

    csv_path = output_dir / "master_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    md_path = output_dir / "master_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {disease_dir.name} architecture comparison\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            formatted = [format_number(row.get(col)) for col in columns]
            f.write("| " + " | ".join(formatted) + " |\n")
    return csv_path, md_path


def plot_train_loss(disease_dir: Path, runs):
    output_dir = disease_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    for run in runs:
        history = run["history"]
        metrics = run["metrics"]
        network = metrics.get("network", run["dir"].name.split("_", 1)[0])
        variant = metrics.get("variant", run["dir"].name.rsplit("_", 1)[-1])
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        ax.plot(
            epochs,
            history.get("train_loss", []),
            label=f"{network} {variant}",
            color=NETWORK_COLORS.get(network, "#7f7f7f"),
            linestyle=VARIANT_STYLES.get(variant, "-"),
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(f"{disease_dir.name} training loss by run")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_curves.png", dpi=200)
    plt.close(fig)


def plot_val_mae(disease_dir: Path, runs):
    output_dir = disease_dir / "comparison"
    fig, ax = plt.subplots(figsize=(16, 9))
    for run in runs:
        history = run["history"]
        metrics = run["metrics"]
        network = metrics.get("network", run["dir"].name.split("_", 1)[0])
        variant = metrics.get("variant", run["dir"].name.rsplit("_", 1)[-1])
        epochs = list(range(1, len(history.get("val_mae", [])) + 1))
        ax.plot(
            epochs,
            history.get("val_mae", []),
            label=f"{network} {variant}",
            color=NETWORK_COLORS.get(network, "#7f7f7f"),
            linestyle=VARIANT_STYLES.get(variant, "-"),
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE")
    ax.set_title(f"{disease_dir.name} validation MAE by run")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(output_dir / "val_mae_curves.png", dpi=200)
    plt.close(fig)


def plot_bar_chart(output_path: Path, labels, values, title, ylabel, color_key=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = [NETWORK_COLORS.get(label.split()[0], "#7f7f7f") for label in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4g}", ha="center", va="bottom", fontsize="x-small")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary_bars(disease_dir: Path, rows):
    output_dir = disease_dir / "comparison"
    labels = [f"{row['network']} {row['variant']}" for row in rows]
    plot_bar_chart(
        output_dir / "test_accuracy_bar.png",
        labels,
        [float(row["test_accuracy"]) for row in rows],
        f"{disease_dir.name} test accuracy",
        "Accuracy",
    )
    plot_bar_chart(
        output_dir / "test_mae_bar.png",
        labels,
        [float(row["test_mae"]) for row in rows],
        f"{disease_dir.name} test MAE",
        "MAE",
    )
    plot_bar_chart(
        output_dir / "param_count_bar.png",
        labels,
        [float(row["param_count_total"]) for row in rows],
        f"{disease_dir.name} parameter count",
        "Parameters",
    )
    plot_bar_chart(
        output_dir / "inference_ms_bar.png",
        labels,
        [float(row["inference_ms_per_image"]) for row in rows],
        f"{disease_dir.name} inference ms per image",
        "ms/image",
    )
    plot_bar_chart(
        output_dir / "training_seconds_bar.png",
        labels,
        [float(row["average_training_seconds_per_epoch"]) for row in rows],
        f"{disease_dir.name} average training seconds per epoch",
        "seconds/epoch",
    )


def plot_confusion_matrix_grid(disease_dir: Path, runs):
    output_dir = disease_dir / "comparison"
    cols = 3
    rows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    for idx, run in enumerate(runs):
        metrics = run["metrics"]
        cm = metrics.get("confusion_matrix", [])
        ax = axes[idx]
        ax.imshow(cm if cm else [[0, 0, 0], [0, 0, 0], [0, 0, 0]], cmap="Blues")
        ax.set_title(f"{metrics.get('network', '')} {metrics.get('variant', '')}")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Mild", "Moderate", "Severe"])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Mild", "Moderate", "Severe"])
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontsize="small")
    for ax in axes[len(runs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_grid.png", dpi=200)
    plt.close(fig)


def run_for_disease(disease_name: str):
    disease_dir = ROOT / disease_name
    if not disease_dir.exists():
        raise FileNotFoundError(f"Missing disease directory: {disease_dir}")
    runs = discover_runs(disease_dir)
    if not runs:
        raise FileNotFoundError(f"No completed run JSON files found under {disease_dir}")
    rows = build_master_rows(disease_dir)
    output_dir = disease_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_master_table(rows, disease_dir)
    plot_train_loss(disease_dir, runs)
    plot_val_mae(disease_dir, runs)
    plot_summary_bars(disease_dir, rows)
    plot_confusion_matrix_grid(disease_dir, runs)
    return output_dir


def main():
    for disease_name in sorted([p.name for p in ROOT.iterdir() if p.is_dir()]):
        output_dir = run_for_disease(disease_name)
        print(f"Saved {disease_name} comparison results to {output_dir}")


if __name__ == "__main__":
    main()
