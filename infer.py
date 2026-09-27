"""CLI inference using ordinal severity models selected for accuracy and MAE.

Acne ordinal: 88.53% accuracy, MAE 0.118 versus categorical 82.08%, 0.190.
Eczema ordinal: 69.69% accuracy, MAE 0.314 versus categorical 64.46%, 0.387.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Global configuration matching training
STAGE1_CLASSES = ["Acne", "Eczema", "Normal", "Other"]
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(ckpt_path: Path, num_classes: int, device: torch.device, ordinal: bool = False):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint.get("model_name", checkpoint.get("arch", "resnet50")).lower()

    if "resnet" in model_name:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1 if ordinal else num_classes)
    elif "efficientnet" in model_name:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1 if ordinal else num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class SkinDiagnosticPipeline:
    def __init__(
        self,
        stage1_path: str = r"runs\resnet50_v2\best_model.pt",
        acne_severity_path: str = r"runs\severity_acne_ordinal\best_model.pt",
        eczema_severity_path: str = r"runs\severity_eczema_ordinal\best_model.pt",
        device: str = "auto",
    ):
        base_dir = Path(__file__).resolve().parent
        self.stage1_path = (base_dir / stage1_path).resolve()
        self.acne_path = (base_dir / acne_severity_path).resolve()
        self.eczema_path = (base_dir / eczema_severity_path).resolve()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.transforms = get_eval_transforms()

        # Stage 1 now uses the four-class model with Other for non-skin rejection: the prior
        # three-class model called a text-table screenshot Eczema (94.41%); v2 calls it Other
        # (99.915%) and calls a cardboard box Other (99.996%).
        # Ordinal models won for both diseases: Acne 88.53% / MAE 0.118 vs 82.08% / 0.190,
        # and Eczema 69.69% / 0.314 vs 64.46% / 0.387 for categorical models.
        print(f"Loading Stage 1 Disease Classifier from: {self.stage1_path.name}")
        self.stage1_model = load_model(self.stage1_path, len(STAGE1_CLASSES), self.device)

        print(f"Loading Stage 2 Acne Severity Assessor from: {self.acne_path.name}")
        self.acne_model = load_model(self.acne_path, len(SEVERITY_CLASSES), self.device, ordinal=True)

        print(f"Loading Stage 2 Eczema Severity Assessor from: {self.eczema_path.name}")
        self.eczema_model = load_model(self.eczema_path, len(SEVERITY_CLASSES), self.device, ordinal=True)
        print("All models successfully loaded onto device:", self.device)

    def predict(self, image_path: str) -> dict:
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image does not exist: {img_p}")

        raw_image = Image.open(img_p).convert("RGB")
        tensor = self.transforms(raw_image).unsqueeze(0).to(self.device)

        # STAGE 1: Disease Classification
        with torch.no_grad():
            stage1_logits = self.stage1_model(tensor)
            stage1_probs = torch.softmax(stage1_logits, dim=1)[0].cpu().numpy()

        disease_idx = int(stage1_probs.argmax())
        predicted_disease = STAGE1_CLASSES[disease_idx]
        disease_confidence = float(stage1_probs[disease_idx])

        disease_breakdown = {
            STAGE1_CLASSES[i]: float(stage1_probs[i]) for i in range(len(STAGE1_CLASSES))
        }

        # STAGE 2: Severity Assessment
        if predicted_disease in {"Normal", "Other"}:
            predicted_severity = "N/A"
            severity_confidence = 1.0
            severity_details = {"N/A": 1.0}
            if predicted_disease == "Normal":
                clinical_status = "Healthy skin - No lesion requiring severity staging"
            else:
                clinical_status = "Other or non-skin input - severity assessment not applicable"
        elif predicted_disease in {"Acne", "Eczema"}:
            severity_model = self.acne_model if predicted_disease == "Acne" else self.eczema_model
            with torch.no_grad():
                raw_severity = float(severity_model(tensor).squeeze().cpu().item())

            rounded_severity = round(raw_severity)
            sev_idx = int(max(0, min(2, rounded_severity)))
            predicted_severity = SEVERITY_CLASSES[sev_idx]
            severity_confidence = float(max(0.0, min(1.0, 1.0 - abs(raw_severity - rounded_severity))))
            severity_details = {
                "raw_value": raw_severity,
                "approximate_certainty": severity_confidence,
            }
            clinical_status = f"{predicted_disease} detected at {predicted_severity.upper()} severity"
        else:
            raise ValueError(f"Unknown disease: {predicted_disease}")

        return {
            "image_path": str(img_p.resolve()),
            "disease": predicted_disease,
            "disease_confidence": disease_confidence,
            "disease_probabilities": disease_breakdown,
            "severity": predicted_severity,
            "severity_confidence": severity_confidence,
            "severity_details": severity_details,
            "clinical_summary": clinical_status,
        }


def format_cli_output(result: dict):
    print("\n" + "=" * 65)
    print("        SKIN DISEASE DETECTION & SEVERITY ASSESSMENT")
    print("=" * 65)
    print(f"Image File          : {Path(result['image_path']).name}")
    print(f"Device              : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 65)
    print("STAGE 1: DISEASE IDENTIFICATION")
    print(f"  Diagnosis         : {result['disease'].upper()}")
    print(f"  Confidence        : {result['disease_confidence']*100:.2f}%")
    print("  Probabilities     :")
    for dis, prob in result["disease_probabilities"].items():
        bar = "#" * int(prob * 20)
        print(f"    - {dis:10s} : {prob*100:6.2f}%  |{bar:<20}|")

    print("-" * 65)
    print("STAGE 2: SEVERITY ASSESSMENT")
    print(f"  Severity Level    : {result['severity'].upper()}")
    if result["severity"] != "N/A":
        print(f"  Approx. certainty : {result['severity_confidence']*100:.2f}%")
        print("  Note              : Not a calibrated probability")
        print(f"  Raw ordinal value : {result['severity_details']['raw_value']:.4f}")
    else:
        print("  Confidence        : 100.00% (Not applicable for healthy skin)")

    print("-" * 65)
    print(f"CLINICAL SUMMARY    : {result['clinical_summary']}")
    print("=" * 65 + "\n")


def run_demo():
    print("\nRunning Automated Demo across Unseen Test Images...")
    base_dir = Path(__file__).resolve().parent
    test_manifest = base_dir / "data" / "dataset_manifest.csv"

    pipeline = SkinDiagnosticPipeline()

    if test_manifest.exists():
        import pandas as pd
        df = pd.read_csv(test_manifest)
        test_df = df[df["split"] == "test"]

        # Pick 1 sample for each major category from the held-out test split
        demo_samples = []
        for cat in ["Acne_Mild", "Acne_Moderate", "Acne_Severe", "Eczema_Mild", "Eczema_Moderate", "Eczema_Severe", "Normal"]:
            cat_rows = test_df[test_df["category"] == cat]
            if len(cat_rows) > 0:
                rel_path = cat_rows.iloc[0]["path"]
                full_p = base_dir / "data" / rel_path
                demo_samples.append((cat, full_p))

        for ground_truth, p in demo_samples:
            print(f">>> Testing Image [Ground Truth: {ground_truth}]:")
            res = pipeline.predict(str(p))
            format_cli_output(res)
    else:
        print("Manifest not found. Please provide an image path directly.")


def main():
    parser = argparse.ArgumentParser(description="Skin Disease & Severity Inference Pipeline")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input skin image")
    parser.add_argument("--demo", action="store_true", help="Run automated demo on test images")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--stage1-ckpt", default=r"runs\resnet50_v2\best_model.pt", help="Stage 1 four-class checkpoint")
    parser.add_argument("--acne-ckpt", default=r"runs\severity_acne_ordinal\best_model.pt", help="Stage 2 Acne ordinal checkpoint")
    parser.add_argument("--eczema-ckpt", default=r"runs\severity_eczema_ordinal\best_model.pt", help="Stage 2 Eczema ordinal checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.image_path:
        print("Usage: python infer.py <path_to_image> [--json] [--demo]")
        print("Example: python infer.py data/splits/test/Acne_Severe/some_image.jpg")
        print("Tip: Use --demo to test automatically across unseen test samples.")
        sys.exit(1)

    pipeline = SkinDiagnosticPipeline(
        stage1_path=args.stage1_ckpt,
        acne_severity_path=args.acne_ckpt,
        eczema_severity_path=args.eczema_ckpt,
        device=args.device,
    )

    result = pipeline.predict(args.image_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        format_cli_output(result)


if __name__ == "__main__":
    main()
