"""
Task 2: Diagnose the weak Eczema severity model.
Outputs:
  a) Full confusion matrix (printed + saved heatmap)
  b) Per-class precision/recall/F1
  c) MAE treating Mild=0, Moderate=1, Severe=2
  d) Grid of 8-10 misclassified examples with true/pred label and confidence
  e) Summary: adjacent vs severe confusion pattern
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_severity_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SEVERITY_CLASSES))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    manifest_path = data_dir / "dataset_manifest.csv"
    ckpt_path = base_dir / "runs" / "severity_eczema" / "best_model.pt"
    out_dir = base_dir / "runs" / "severity_eczema"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Eczema severity model from {ckpt_path.name}...")
    model = load_severity_model(ckpt_path, device)

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    df = pd.read_csv(manifest_path)
    # Filter to Eczema test samples with valid severity labels
    eczema_test = df[
        (df["split"] == "test") &
        (df["disease"] == "Eczema") &
        (df["severity"].isin(SEVERITY_CLASSES))
    ].copy().reset_index(drop=True)
    print(f"Eczema test samples: {len(eczema_test)}")

    sev_to_idx = {name: i for i, name in enumerate(SEVERITY_CLASSES)}

    y_true = []
    y_pred = []
    y_pred_conf = []
    y_pred_probs = []
    image_paths = []
    filenames = []

    for idx, row in eczema_test.iterrows():
        rel_path = row["path"]
        img_path = data_dir / rel_path
        if not img_path.exists():
            print(f"  WARNING: Missing image: {img_path}")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        tensor = eval_tfms(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        true_idx = sev_to_idx[row["severity"]]

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_pred_conf.append(float(probs[pred_idx]))
        y_pred_probs.append(probs)
        image_paths.append(img_path)
        filenames.append(row["filename"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ======== (a) Confusion Matrix ========
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print("(a) CONFUSION MATRIX")
    print("=" * 60)
    print(f"{'':15s} {'Pred Mild':>10s} {'Pred Mod':>10s} {'Pred Sev':>10s}")
    for i, row_name in enumerate(SEVERITY_CLASSES):
        print(f"{'True ' + row_name:15s} {cm[i,0]:10d} {cm[i,1]:10d} {cm[i,2]:10d}")

    # Save heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SEVERITY_CLASSES,
        yticklabels=SEVERITY_CLASSES,
        ax=ax,
    )
    ax.set_title("Eczema Severity Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Severity", fontsize=12)
    ax.set_ylabel("Actual Severity", fontsize=12)
    plt.tight_layout()
    cm_path = out_dir / "eczema_severity_diagnosis_cm.png"
    plt.savefig(cm_path, dpi=250)
    plt.close()
    print(f"Saved: {cm_path}")

    # ======== (b) Per-class Precision/Recall/F1 ========
    print("\n" + "=" * 60)
    print("(b) PER-CLASS METRICS")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=SEVERITY_CLASSES,
        digits=4,
        zero_division=0,
    )
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")

    # ======== (c) Mean Absolute Error (ordinal) ========
    mae = np.mean(np.abs(y_true - y_pred))
    print("\n" + "=" * 60)
    print("(c) ORDINAL MEAN ABSOLUTE ERROR (MAE)")
    print("=" * 60)
    print(f"MAE (Mild=0, Moderate=1, Severe=2): {mae:.4f}")
    print(f"Interpretation: On average, predictions are off by {mae:.2f} severity grades.")

    # Error distance distribution
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    exact_match = np.sum(abs_errors == 0)
    off_by_one = np.sum(abs_errors == 1)
    off_by_two = np.sum(abs_errors == 2)
    total = len(abs_errors)

    print(f"\nError Distance Distribution:")
    print(f"  Exact match (0 grades off): {exact_match:4d} / {total} ({exact_match/total*100:.1f}%)")
    print(f"  Adjacent error (1 grade off): {off_by_one:4d} / {total} ({off_by_one/total*100:.1f}%)")
    print(f"  Severe error (2 grades off):  {off_by_two:4d} / {total} ({off_by_two/total*100:.1f}%)")

    # Directional error analysis
    under_predict = np.sum(errors < 0)  # pred < true (predicting milder than actual)
    over_predict = np.sum(errors > 0)   # pred > true (predicting more severe than actual)
    print(f"\nDirectional Bias:")
    print(f"  Under-predicting (pred milder than truth): {under_predict}")
    print(f"  Over-predicting (pred more severe than truth): {over_predict}")

    # Specific confusion patterns
    mild_as_mod = cm[0, 1]
    mod_as_mild = cm[1, 0]
    mod_as_sev = cm[1, 2]
    sev_as_mod = cm[2, 1]
    mild_as_sev = cm[0, 2]
    sev_as_mild = cm[2, 0]

    print(f"\nAdjacent Confusions:")
    print(f"  Mild -> Moderate: {mild_as_mod}")
    print(f"  Moderate -> Mild: {mod_as_mild}")
    print(f"  Moderate -> Severe: {mod_as_sev}")
    print(f"  Severe -> Moderate: {sev_as_mod}")
    print(f"Non-adjacent (Severe) Confusions:")
    print(f"  Mild -> Severe: {mild_as_sev}")
    print(f"  Severe -> Mild: {sev_as_mild}")

    adjacent_total = mild_as_mod + mod_as_mild + mod_as_sev + sev_as_mod
    severe_total = mild_as_sev + sev_as_mild
    all_errors = adjacent_total + severe_total
    print(f"\nSummary: {adjacent_total}/{all_errors} errors ({adjacent_total/max(all_errors,1)*100:.1f}%) are adjacent-class confusions")
    print(f"         {severe_total}/{all_errors} errors ({severe_total/max(all_errors,1)*100:.1f}%) are severe (2-grade) confusions")

    # ======== (d) Grid of Misclassified Examples ========
    print("\n" + "=" * 60)
    print("(d) MISCLASSIFIED EXAMPLES")
    print("=" * 60)

    misclass_indices = np.where(y_true != y_pred)[0]
    print(f"Total misclassified: {len(misclass_indices)}")

    # Sort by confidence (highest confidence errors are most concerning)
    misclass_data = []
    for mi in misclass_indices:
        misclass_data.append({
            "idx": mi,
            "filename": filenames[mi],
            "path": image_paths[mi],
            "true_sev": SEVERITY_CLASSES[y_true[mi]],
            "pred_sev": SEVERITY_CLASSES[y_pred[mi]],
            "confidence": y_pred_conf[mi],
            "true_idx": y_true[mi],
            "pred_idx": y_pred[mi],
            "error_dist": abs(int(y_true[mi]) - int(y_pred[mi])),
        })

    # Sort by confidence descending (most confident errors first)
    misclass_data.sort(key=lambda x: -x["confidence"])

    # Show top 10
    display_count = min(10, len(misclass_data))
    print(f"\nTop {display_count} highest-confidence misclassifications:")
    for m in misclass_data[:display_count]:
        print(f"  {m['filename'][:40]:40s} | True: {m['true_sev']:8s} | Pred: {m['pred_sev']:8s} | Conf: {m['confidence']*100:.1f}% | Off by {m['error_dist']}")

    # Generate image grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()
    for i in range(display_count):
        m = misclass_data[i]
        try:
            pil_img = Image.open(m["path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            axes[i].imshow(pil_img)
        except Exception:
            axes[i].text(0.5, 0.5, "Error", ha="center", va="center")

        err_color = "orange" if m["error_dist"] == 1 else "red"
        axes[i].set_title(
            f"True: {m['true_sev']}\nPred: {m['pred_sev']} ({m['confidence']*100:.1f}%)",
            fontsize=10,
            color=err_color,
            fontweight="bold",
        )
        axes[i].axis("off")

    for j in range(display_count, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Eczema Severity: Top Misclassified Test Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid_path = out_dir / "eczema_severity_misclassified_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"\nSaved misclassified grid: {grid_path}")

    # ======== (e) Final diagnosis summary ========
    print("\n" + "=" * 60)
    print("(e) DIAGNOSIS SUMMARY")
    print("=" * 60)
    if adjacent_total > severe_total * 2:
        print("PATTERN: The model PRIMARILY confuses ADJACENT severity classes.")
        print("This suggests the visual boundary between Mild<->Moderate and")
        print("Moderate<->Severe is ambiguous, which is consistent with clinical")
        print("inter-observer variability in eczema severity grading.")
        print("RECOMMENDATION: Ordinal regression (Task 3) may help by penalizing")
        print("2-grade errors more heavily than 1-grade errors.")
    else:
        print("PATTERN: The model makes a mix of adjacent AND severe errors.")
        print("This suggests fundamental difficulty in the task, possibly due to")
        print("label noise or insufficient visual signal for severity grading.")


if __name__ == "__main__":
    main()
