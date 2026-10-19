"""Aggregate the three recorded severity-model seeds without training."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED_DIRS = {
    42: ROOT / "runs" / "arch_comparison",
    1: ROOT / "runs" / "arch_comparison_seed1",
    2: ROOT / "runs" / "arch_comparison_seed2",
}
MODELS = (
    ("ownnet", "ownnet_final"),
    ("densenet121", "densenet121_baseline"),
    ("resnet50", "resnet50_baseline"),
)
DISEASES = ("Acne", "Eczema")
METRICS = ("test_accuracy", "test_mae")


def metric_path(seed_root: Path, disease: str, run_name: str) -> Path:
    return seed_root / disease / run_name / "metrics.json"


def main() -> None:
    collected = {}
    print("Individual seed results (raw JSON values)")
    for seed, seed_root in SEED_DIRS.items():
        for disease in DISEASES:
            for model, run_name in MODELS:
                path = metric_path(seed_root, disease, run_name)
                key = (seed, disease, model)
                if not path.is_file():
                    print(f"seed={seed} disease={disease} model={model} MISSING: {path.relative_to(ROOT)}")
                    collected[key] = None
                    continue
                with path.open(encoding="utf-8") as handle:
                    metrics = json.load(handle)
                values = {metric: metrics.get(metric) for metric in METRICS}
                collected[key] = values
                print(
                    f"seed={seed} disease={disease} model={model} "
                    f"test_accuracy={values['test_accuracy']!r} "
                    f"test_mae={values['test_mae']!r} "
                    f"file={path.relative_to(ROOT)}"
                )

    rows = []
    print("\nMean +/- sample standard deviation by disease and model")
    for disease in DISEASES:
        for model, _ in MODELS:
            row = {"disease": disease, "model": model}
            for metric in METRICS:
                values = [
                    item[metric]
                    for seed in SEED_DIRS
                    if (item := collected[(seed, disease, model)]) is not None
                    and isinstance(item[metric], (int, float))
                ]
                for seed in SEED_DIRS:
                    seed_result = collected[(seed, disease, model)]
                    row[f"{metric}_seed_{seed}"] = (
                        seed_result[metric]
                        if seed_result is not None and isinstance(seed_result[metric], (int, float))
                        else ""
                    )
                mean = statistics.fmean(values) if values else None
                std = statistics.stdev(values) if len(values) > 1 else None
                row[f"n_{metric}"] = len(values)
                row[f"mean_{metric}"] = mean
                row[f"std_{metric}"] = std
                mean_text = f"{mean:.12g}" if mean is not None else "MISSING"
                std_text = f"{std:.12g}" if std is not None else ("not estimable" if len(values) == 1 else "MISSING")
                print(
                    f"disease={disease} model={model} metric={metric} "
                    f"n={len(values)} mean={mean_text} std={std_text}"
                )
            rows.append(row)

    output = ROOT / "report" / "tables" / "seed_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "disease",
        "model",
        *(f"{metric}_seed_{seed}" for metric in METRICS for seed in SEED_DIRS),
        "n_test_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "n_test_mae",
        "mean_test_mae",
        "std_test_mae",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
