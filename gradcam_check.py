import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models

CLASS_NAMES = ["Acne", "Eczema", "Normal"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Compute weights via global average pooling of gradients
        gradients = self.gradients.detach()  # [1, 2048, 7, 7]
        activations = self.activations.detach()  # [1, 2048, 7, 7]

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, 2048, 1, 1]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, 7, 7]

        # ReLU on combined activations
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, probs[0, target_class].item(), probs[0].detach().cpu().numpy()


def overlay_cam_on_image(img_pil, cam, alpha=0.45):
    # Resize CAM to match original image size
    img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # Resize CAM using PIL
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    cam_np = np.array(cam_pil).astype(np.float32) / 255.0

    # Apply colormap (jet)
    cmap = plt.get_cmap("jet")
    colored_cam = cmap(cam_np)[:, :, :3]  # Drop alpha channel

    # Blend
    overlay = (1.0 - alpha) * img_np + alpha * colored_cam
    overlay = np.clip(overlay, 0.0, 1.0)
    return img_resized, overlay, cam_np


def load_stage1_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    manifest_path = data_dir / "dataset_manifest.csv"
    ckpt_path = base_dir / "runs" / "resnet50" / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Stage 1 ResNet50 from {ckpt_path.name} on {device}...")
    model = load_stage1_model(ckpt_path, device)

    # ResNet50 target layer: last convolutional block of layer4
    target_layer = model.layer4[-1].conv3
    grad_cam = GradCAM(model, target_layer)

    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    print(f"Scanning {len(test_df)} test images for predictions and errors...")

    results = []
    for idx, row in test_df.iterrows():
        rel_path = row["path"]
        img_full_path = data_dir / rel_path
        if not img_full_path.exists():
            continue

        true_disease = row["disease"]
        true_idx = CLASS_NAMES.index(true_disease)

        try:
            pil_img = Image.open(img_full_path).convert("RGB")
        except Exception:
            continue

        input_tensor = eval_tfms(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_disease = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        results.append({
            "idx": idx,
            "filename": row["filename"],
            "path": img_full_path,
            "true_disease": true_disease,
            "true_idx": true_idx,
            "pred_disease": pred_disease,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "correct": (true_idx == pred_idx),
            "probs": probs,
        })

    res_df = pd.DataFrame(results)
    correct_df = res_df[res_df["correct"] == True]
    misclass_df = res_df[res_df["correct"] == False]

    print(f"\n[Test Split Summary]")
    print(f"Total evaluated images : {len(res_df)}")
    print(f"Correctly classified   : {len(correct_df)} ({len(correct_df)/len(res_df)*100:.2f}%)")
    print(f"Misclassified images   : {len(misclass_df)} ({len(misclass_df)/len(res_df)*100:.2f}%)")

    # Select 15 correctly classified: 5 Acne, 5 Eczema, 5 Normal
    selected_correct = []
    for dis in CLASS_NAMES:
        dis_samples = correct_df[correct_df["true_disease"] == dis].sample(
            n=min(5, len(correct_df[correct_df["true_disease"] == dis])), random_state=42
        )
        selected_correct.extend(dis_samples.to_dict("records"))

    # All misclassified images (up to 10)
    selected_misclass = misclass_df.to_dict("records")
    print(f"\nSelected {len(selected_correct)} correct samples (5 per class) and {len(selected_misclass)} misclassified samples.")

    # 1. Generate Grad-CAM Grid for Correctly Classified Images (3 columns: Original, Heatmap, Overlay)
    print("\nGenerating Grad-CAM overlays for correctly classified samples...")
    fig, axes = plt.subplots(len(selected_correct), 3, figsize=(11, len(selected_correct) * 3.0))

    shortcut_analysis = []

    for i, item in enumerate(selected_correct):
        pil_img = Image.open(item["path"]).convert("RGB")
        input_tensor = eval_tfms(pil_img).unsqueeze(0).to(device)

        cam, target_cls, conf, _ = grad_cam.generate(input_tensor, target_class=item["pred_idx"])
        img_resized, overlay, cam_np = overlay_cam_on_image(pil_img, cam)

        # Check where the maximum activation lies (center/skin vs border/peripheral)
        h, w = cam_np.shape
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
        center_mask = ~border_mask

        center_energy = cam_np[center_mask].sum()
        border_energy = cam_np[border_mask].sum()
        focus_ratio = center_energy / (center_energy + border_energy + 1e-8)

        shortcut_analysis.append({
            "filename": item["filename"],
            "true": item["true_disease"],
            "pred": item["pred_disease"],
            "conf": conf,
            "center_focus_ratio": focus_ratio,
        })

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"Original: {item['true_disease']}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_np, cmap="jet")
        axes[i, 1].set_title(f"CAM Heatmap (Pred: {item['pred_disease']})", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay (Conf: {conf*100:.1f}%)", fontsize=10, color="darkgreen")
        axes[i, 2].axis("off")

    plt.tight_layout()
    correct_plot_path = base_dir / "gradcam_correct.png"
    plt.savefig(correct_plot_path, dpi=200)
    plt.close()
    print(f"Saved: {correct_plot_path}")

    # 2. Generate Grad-CAM Grid for Misclassified Images
    if len(selected_misclass) > 0:
        print("\nGenerating Grad-CAM overlays for misclassified samples...")
        fig, axes = plt.subplots(len(selected_misclass), 3, figsize=(11, len(selected_misclass) * 3.2))
        if len(selected_misclass) == 1:
            axes = np.expand_dims(axes, 0)

        for i, item in enumerate(selected_misclass):
            pil_img = Image.open(item["path"]).convert("RGB")
            input_tensor = eval_tfms(pil_img).unsqueeze(0).to(device)

            cam, target_cls, conf, probs = grad_cam.generate(input_tensor, target_class=item["pred_idx"])
            img_resized, overlay, cam_np = overlay_cam_on_image(pil_img, cam)

            axes[i, 0].imshow(img_resized)
            axes[i, 0].set_title(f"Original (True: {item['true_disease']})", fontsize=10, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cam_np, cmap="jet")
            axes[i, 1].set_title(f"CAM for Predicted: {item['pred_disease']}", fontsize=10)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"MISCLASSIFIED: {item['pred_disease']} ({conf*100:.1f}%)\nTrue: {item['true_disease']}", fontsize=10, color="crimson")
            axes[i, 2].axis("off")

        plt.tight_layout()
        misclass_plot_path = base_dir / "gradcam_misclassified.png"
        plt.savefig(misclass_plot_path, dpi=200)
        plt.close()
        print(f"Saved: {misclass_plot_path}")

    # Also save a unified summary grid (e.g. gradcam_results.png) with representative images
    print("\nGenerating unified gradcam_results.png...")
    rep_samples = selected_correct[:6] + selected_misclass[:3]
    fig, axes = plt.subplots(len(rep_samples), 3, figsize=(11, len(rep_samples) * 3.0))

    for i, item in enumerate(rep_samples):
        pil_img = Image.open(item["path"]).convert("RGB")
        input_tensor = eval_tfms(pil_img).unsqueeze(0).to(device)

        cam, target_cls, conf, _ = grad_cam.generate(input_tensor, target_class=item["pred_idx"])
        img_resized, overlay, cam_np = overlay_cam_on_image(pil_img, cam)

        title_color = "darkgreen" if item["correct"] else "crimson"
        status_tag = "CORRECT" if item["correct"] else "MISCLASSIFIED"

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"True: {item['true_disease']}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_np, cmap="jet")
        axes[i, 1].set_title(f"CAM: {item['pred_disease']}", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"[{status_tag}] Pred: {item['pred_disease']} ({conf*100:.1f}%)", fontsize=10, color=title_color)
        axes[i, 2].axis("off")

    plt.tight_layout()
    unified_plot_path = base_dir / "gradcam_results.png"
    plt.savefig(unified_plot_path, dpi=200)
    plt.close()
    print(f"Saved: {unified_plot_path}")

    # Print shortcut analysis
    print("\n" + "=" * 65)
    print("SHORTCUT LEARNING / FOCUS AUDIT (Correct Samples):")
    print("=" * 65)
    for sa in shortcut_analysis:
        print(f"  [{sa['true']:6s}] {sa['filename'][:30]:30s} -> Center Focus: {sa['center_focus_ratio']*100:.1f}% | Conf: {sa['conf']*100:.1f}%")

    avg_focus = np.mean([sa['center_focus_ratio'] for sa in shortcut_analysis])
    print(f"\nMean Central/Lesion Region Activation Ratio: {avg_focus*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
