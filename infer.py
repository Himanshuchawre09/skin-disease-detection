"""CLI inference for seven disease classes and categorical Acne/Eczema severity."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Global configuration matching training
STAGE1_CLASSES = ["Acne", "Eczema", "Normal", "Other", "NV", "MEL", "BKL"]
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


def load_model(
    ckpt_path: Path,
    num_classes: int,
    device: torch.device,
    expected_classes: list[str] | None = None,
):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint.get("model_name", checkpoint.get("arch", "resnet50")).lower()
    if expected_classes is not None:
        checkpoint_classes = checkpoint.get("class_names", checkpoint.get("severity_classes"))
        if checkpoint_classes != expected_classes:
            raise ValueError(
                f"Unexpected class order in {ckpt_path}: {checkpoint_classes}; "
                f"expected {expected_classes}"
            )
    checkpoint_image_size = checkpoint.get("image_size", IMAGE_SIZE)
    if checkpoint_image_size != IMAGE_SIZE:
        raise ValueError(
            f"Unexpected image size in {ckpt_path}: {checkpoint_image_size}; "
            f"expected {IMAGE_SIZE}"
        )

    if "resnet" in model_name:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif "efficientnet" in model_name:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
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
        stage1_path: str = r"runs\resnet50_v3\best_model.pt",
        acne_severity_path: str = r"runs\severity_acne\best_model.pt",
        eczema_severity_path: str = r"runs\severity_eczema\best_model.pt",
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

        # Stage 1 uses the merged seven-class model. Only Acne and Eczema have severity heads.
        print(f"Loading Stage 1 Disease Classifier from: {self.stage1_path.name}")
        self.stage1_model = load_model(
            self.stage1_path,
            len(STAGE1_CLASSES),
            self.device,
            expected_classes=STAGE1_CLASSES,
        )

        print(f"Loading Stage 2 Acne Severity Assessor from: {self.acne_path.name}")
        self.acne_model = load_model(
            self.acne_path,
            len(SEVERITY_CLASSES),
            self.device,
            expected_classes=SEVERITY_CLASSES,
        )

        print(f"Loading Stage 2 Eczema Severity Assessor from: {self.eczema_path.name}")
        self.eczema_model = load_model(
            self.eczema_path,
            len(SEVERITY_CLASSES),
            self.device,
            expected_classes=SEVERITY_CLASSES,
        )
        print("All models successfully loaded onto device:", self.device)

    def predict(self, image_path: str) -> dict:
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image does not exist: {img_p}")

        with Image.open(img_p) as image:
            raw_image = image.convert("RGB")
        return self.predict_image(raw_image, str(img_p.resolve()))

    def predict_image(self, raw_image: Image.Image, image_path: str = "uploaded image") -> dict:
        raw_image = raw_image.convert("RGB")
        tensor = self.transforms(raw_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            stage1_logits = self.stage1_model(tensor)
            stage1_probs = torch.softmax(stage1_logits, dim=1)[0].cpu()

        disease_idx = int(stage1_probs.argmax())
        predicted_disease = STAGE1_CLASSES[disease_idx]
        disease_confidence = float(stage1_probs[disease_idx])
        disease_breakdown = {
            name: float(stage1_probs[index])
            for index, name in enumerate(STAGE1_CLASSES)
        }

        if predicted_disease in {"Acne", "Eczema"}:
            severity_model = self.acne_model if predicted_disease == "Acne" else self.eczema_model
            with torch.inference_mode():
                severity_probs = torch.softmax(severity_model(tensor), dim=1)[0].cpu()

            severity_idx = int(severity_probs.argmax())
            predicted_severity = SEVERITY_CLASSES[severity_idx]
            severity_confidence = float(severity_probs[severity_idx])
            severity_details = {
                "probabilities": {
                    name: float(severity_probs[index])
                    for index, name in enumerate(SEVERITY_CLASSES)
                }
            }
            clinical_status = f"{predicted_disease} detected at {predicted_severity.upper()} severity"
        elif predicted_disease == "Normal":
            predicted_severity = "N/A"
            severity_confidence = 1.0
            severity_details = {"N/A": 1.0}
            clinical_status = "Healthy skin - No lesion requiring severity staging"
        elif predicted_disease == "Other":
            predicted_severity = "N/A"
            severity_confidence = 1.0
            severity_details = {"N/A": 1.0}
            clinical_status = "Other or non-skin input - severity assessment not applicable"
        elif predicted_disease in {"NV", "MEL", "BKL"}:
            predicted_severity = "N/A"
            severity_confidence = 1.0
            severity_details = {"N/A": 1.0}
            clinical_status = (
                f"{predicted_disease} lesion class - "
                "severity assessment not applicable"
            )
        else:
            raise ValueError(f"Unknown disease: {predicted_disease}")

        return {
            "image_path": image_path,
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
        print(f"  Confidence        : {result['severity_confidence']*100:.2f}%")
        print("  Note              : Not a calibrated probability")
    else:
        print("  Confidence        : N/A (severity assessment not applicable)")

    print("-" * 65)
    print(f"CLINICAL SUMMARY    : {result['clinical_summary']}")
    print("=" * 65 + "\n")


def run_demo():
    print("\nRunning Automated Demo across Unseen Test Images...")
    base_dir = Path(__file__).resolve().parent
    test_manifest = base_dir / "data" / "dataset_manifest_v3.csv"

    pipeline = SkinDiagnosticPipeline()

    if test_manifest.exists():
        import pandas as pd
        df = pd.read_csv(test_manifest)
        test_df = df[df["split"] == "test"]

        # Pick one sample per Stage 1 class from the held-out test split.
        demo_samples = []
        for class_name in STAGE1_CLASSES:
            class_rows = test_df[test_df["disease"] == class_name]
            if not class_rows.empty:
                rel_path = str(class_rows.iloc[0]["path"]).replace("\\", "/")
                full_p = base_dir / "data" / rel_path
                demo_samples.append((class_name, full_p))

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
    parser.add_argument("--stage1-ckpt", default=r"runs\resnet50_v3\best_model.pt", help="Stage 1 seven-class checkpoint")
    parser.add_argument("--acne-ckpt", default=r"runs\severity_acne\best_model.pt", help="Categorical Acne severity checkpoint")
    parser.add_argument("--eczema-ckpt", default=r"runs\severity_eczema\best_model.pt", help="Categorical Eczema severity checkpoint")
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
