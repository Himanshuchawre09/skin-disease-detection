"""Validate infer.py across every image in the held-out test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from infer import SkinDiagnosticPipeline


SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
DISEASE_CLASSES = ["Acne", "Eczema", "Normal", "Other"]


def main():
    parser = argparse.ArgumentParser(description="Inference-only validation on the test split")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manifest", default="data/dataset_manifest_v2.csv")
    parser.add_argument("--batch-report", default="runs/test_inference_validation_v2.json")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    manifest_path = (base_dir / args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    test_df = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    pipeline = SkinDiagnosticPipeline(device=args.device)

    errors = []
    predictions = []
    non_severity_violations = []
    ground_truth_non_severity_misroutes = []
    routing_calls = {"Acne": 0, "Eczema": 0}

    def count_stage2_call(disease):
        def hook(_module, _inputs, _output):
            routing_calls[disease] += 1
        return hook

    hook_handles = [
        pipeline.acne_model.register_forward_hook(count_stage2_call("Acne")),
        pipeline.eczema_model.register_forward_hook(count_stage2_call("Eczema")),
    ]

    for _, row in test_df.iterrows():
        image_path = (base_dir / "data" / str(row["path"])).resolve()
        record = {
            "filename": str(row["filename"]),
            "path": str(image_path),
            "category": str(row["category"]),
            "true_disease": str(row["disease"]),
            "true_severity": str(row["severity"]),
        }
        try:
            before_calls = routing_calls.copy()
            result = pipeline.predict(str(image_path))
            calls_during_routing = {
                disease: routing_calls[disease] - before_calls[disease]
                for disease in routing_calls
            }
            record.update(
                {
                    "predicted_disease": result["disease"],
                    "predicted_severity": result["severity"],
                    "disease_correct": result["disease"] == row["disease"],
                    "severity_model_calls": calls_during_routing,
                }
            )
            if row["disease"] in {"Normal", "Other"}:
                record["severity_correct"] = result["severity"] == "N/A"
                if result["disease"] in {"Normal", "Other"}:
                    if result["severity"] != "N/A" or sum(calls_during_routing.values()) != 0:
                        non_severity_violations.append(record.copy())
                else:
                    ground_truth_non_severity_misroutes.append(record.copy())
            else:
                transform_tensor = pipeline.transforms(Image.open(image_path).convert("RGB"))
                transform_tensor = transform_tensor.unsqueeze(0).to(pipeline.device)
                severity_model = pipeline.acne_model if row["disease"] == "Acne" else pipeline.eczema_model
                with torch.inference_mode():
                    raw_severity = float(severity_model(transform_tensor).squeeze().cpu().item())
                direct_severity = SEVERITY_CLASSES[int(np.clip(round(raw_severity), 0, 2))]
                record["direct_stage2_severity"] = direct_severity
                record["direct_stage2_correct"] = direct_severity == row["severity"]
                record["end_to_end_severity_correct"] = (
                    result["disease"] == row["disease"]
                    and result["severity"] == row["severity"]
                )
                record["severity_correct"] = record["end_to_end_severity_correct"]
            predictions.append(record)
        except Exception as exc:
            record.update({"error_type": type(exc).__name__, "error": str(exc)})
            errors.append(record)

    for handle in hook_handles:
        handle.remove()

    completed = len(predictions)
    total = len(test_df)
    disease_correct = sum(item["disease_correct"] for item in predictions)
    disease_accuracy = disease_correct / completed if completed else 0.0

    disease_counts = {}
    for disease in DISEASE_CLASSES:
        disease_items = [item for item in predictions if item["true_disease"] == disease]
        disease_counts[disease] = {
            "total": len(disease_items),
            "correct": sum(item["disease_correct"] for item in disease_items),
            "accuracy": (
                sum(item["disease_correct"] for item in disease_items) / len(disease_items)
                if disease_items
                else 0.0
            ),
        }

    severity_counts = {}
    for disease in ["Acne", "Eczema"]:
        all_items = [item for item in predictions if item["true_disease"] == disease]
        correctly_routed = [item for item in all_items if item["predicted_disease"] == disease]
        severity_counts[disease] = {
            "all_true_disease_images": len(all_items),
            "direct_stage2_correct": sum(item.get("direct_stage2_correct", False) for item in all_items),
            "direct_stage2_accuracy": (
                sum(item.get("direct_stage2_correct", False) for item in all_items) / len(all_items)
                if all_items
                else 0.0
            ),
            "end_to_end_correct": sum(item.get("end_to_end_severity_correct", False) for item in all_items),
            "end_to_end_accuracy": (
                sum(item.get("end_to_end_severity_correct", False) for item in all_items) / len(all_items)
                if all_items
                else 0.0
            ),
            "correctly_routed_to_severity_model": len(correctly_routed),
            "conditional_severity_correct": sum(item.get("direct_stage2_correct", False) for item in correctly_routed),
            "conditional_severity_accuracy": (
                sum(item.get("direct_stage2_correct", False) for item in correctly_routed) / len(correctly_routed)
                if correctly_routed
                else 0.0
            ),
        }

    output = {
        "manifest": str(manifest_path),
        "total_test_images": total,
        "completed_predictions": completed,
        "errors_count": len(errors),
        "errors": errors,
        "disease_accuracy": disease_accuracy,
        "disease_correct": disease_correct,
        "disease_counts": disease_counts,
        "severity_counts": severity_counts,
        "non_severity_classes": {
            disease: {
                "predicted_count": sum(item["predicted_disease"] == disease for item in predictions),
                "violations": sum(
                    item["predicted_disease"] == disease and (
                        item["predicted_severity"] != "N/A"
                        or sum(item["severity_model_calls"].values()) != 0
                    )
                    for item in predictions
                ),
            }
            for disease in ["Normal", "Other"]
        },
        "non_severity_violations": non_severity_violations,
        "ground_truth_non_severity_misroutes": ground_truth_non_severity_misroutes,
    }
    output_path = (base_dir / args.batch_report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("TEST-SPLIT INFERENCE VALIDATION")
    print(f"Total test images: {total}")
    print(f"Completed predictions: {completed}")
    print(f"Errors/crashes: {len(errors)}")
    print(f"Disease accuracy: {disease_correct}/{completed} = {disease_accuracy:.4%}")
    for disease, metrics in severity_counts.items():
        print(
            f"{disease} severity accuracy: "
            f"direct Stage 2 {metrics['direct_stage2_correct']}/{metrics['all_true_disease_images']} "
            f"= {metrics['direct_stage2_accuracy']:.4%}; "
            f"end-to-end {metrics['end_to_end_correct']}/{metrics['all_true_disease_images']} "
            f"= {metrics['end_to_end_accuracy']:.4%}; "
            f"conditional {metrics['conditional_severity_correct']}/"
            f"{metrics['correctly_routed_to_severity_model']} "
            f"= {metrics['conditional_severity_accuracy']:.4%}"
        )
    for disease, metrics in output["non_severity_classes"].items():
        print(
            f"Predicted {disease} images triggering severity: "
            f"{metrics['violations']}/{metrics['predicted_count']}"
        )
    print(
        "Ground-truth Normal/Other images misrouted to Acne/Eczema: "
        f"{len(ground_truth_non_severity_misroutes)}"
    )
    for item in ground_truth_non_severity_misroutes:
        print(
            f"- {item['filename']}: true={item['true_disease']}, "
            f"predicted={item['predicted_disease']}"
        )
    if errors:
        print("ERROR DETAILS:")
        for error in errors:
            print(f"- {error['path']}: {error['error_type']}: {error['error']}")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
