"""Create a lesion-grouped HAM10000 manifest for nv, mel, and bkl only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
CLASS_CODES = ["nv", "mel", "bkl"]
SPLIT_ORDER = ["train", "val", "test"]
N_SPLITS = 20


def main():
    data_root = Path(__file__).resolve().parent / "data"
    metadata_path = data_root / "HAM10000_metadata.csv"
    output_dir = data_root / "ham10000_3class"
    manifest_path = output_dir / "split_manifest.csv"
    summary_path = output_dir / "split_summary.json"

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {output_dir}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"HAM10000 metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    selected = metadata[metadata["dx"].isin(CLASS_CODES)].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No rows found for classes: {CLASS_CODES}")
    if selected["lesion_id"].isna().any():
        raise ValueError("Selected metadata contains missing lesion_id values")
    if selected["image_id"].duplicated().any():
        raise ValueError("Selected metadata contains duplicate image_id values")

    lesion_class_counts = selected.groupby("lesion_id")["dx"].nunique()
    mixed_label_lesions = lesion_class_counts[lesion_class_counts > 1]
    if not mixed_label_lesions.empty:
        raise ValueError(
            f"Found lesions assigned multiple dx labels: {len(mixed_label_lesions)}"
        )

    image_paths = []
    for image_id in selected["image_id"].astype(str):
        relative_path = None
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            candidate = Path(part) / f"{image_id}.jpg"
            if (data_root / candidate).is_file():
                relative_path = candidate.as_posix()
                break
        if relative_path is None:
            raise FileNotFoundError(f"No extracted JPEG found for image_id {image_id}")
        image_paths.append(relative_path)
    selected["path"] = image_paths

    fold_ids = np.full(len(selected), -1, dtype=np.int16)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    for fold_id, (_, held_out_indices) in enumerate(
        splitter.split(
            X=np.zeros(len(selected)),
            y=selected["dx"].to_numpy(),
            groups=selected["lesion_id"].to_numpy(),
        )
    ):
        fold_ids[held_out_indices] = fold_id
    if (fold_ids < 0).any():
        raise RuntimeError("Some images were not assigned to a stratified group fold")

    selected["split"] = np.select(
        [fold_ids < 3, fold_ids < 6],
        ["test", "val"],
        default="train",
    )

    lesion_sets = {
        split: set(selected.loc[selected["split"] == split, "lesion_id"])
        for split in SPLIT_ORDER
    }
    overlap_counts = {
        f"{first}_vs_{second}": len(lesion_sets[first] & lesion_sets[second])
        for index, first in enumerate(SPLIT_ORDER)
        for second in SPLIT_ORDER[index + 1 :]
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"Lesion leakage detected: {overlap_counts}")

    class_counts = pd.crosstab(selected["dx"], selected["split"])
    class_counts = class_counts.reindex(
        index=CLASS_CODES,
        columns=SPLIT_ORDER,
        fill_value=0,
    )
    split_image_counts = selected["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    split_lesion_counts = (
        selected.groupby("split")["lesion_id"].nunique().reindex(SPLIT_ORDER, fill_value=0)
    )

    summary = {
        "source_metadata": "data/HAM10000_metadata.csv",
        "image_root": "data",
        "selected_dx": CLASS_CODES,
        "dx_names": {
            "nv": "melanocytic nevus",
            "mel": "melanoma",
            "bkl": "benign keratosis-like lesions",
        },
        "seed": SEED,
        "split_method": "StratifiedGroupKFold grouped by lesion_id; 20 folds",
        "fold_assignment": "folds 0-2=test, 3-5=val, 6-19=train",
        "target_split_proportions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_images": int(len(selected)),
        "total_lesions": int(selected["lesion_id"].nunique()),
        "split_image_counts": {key: int(value) for key, value in split_image_counts.items()},
        "split_lesion_counts": {key: int(value) for key, value in split_lesion_counts.items()},
        "split_image_proportions": {
            key: float(split_image_counts[key] / len(selected)) for key in SPLIT_ORDER
        },
        "class_image_counts_by_split": {
            dx: {split: int(class_counts.loc[dx, split]) for split in SPLIT_ORDER}
            for dx in CLASS_CODES
        },
        "lesion_id_overlap_counts": overlap_counts,
        "metadata_rows_without_image": 0,
    }

    output_dir.mkdir(parents=False, exist_ok=False)
    ordered_columns = list(metadata.columns) + ["path", "split"]
    selected[ordered_columns].to_csv(manifest_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("HAM10000 three-class split prepared (no model training performed)")
    print(f"Images: {len(selected)} | lesion_id groups: {selected['lesion_id'].nunique()}")
    print("Image counts by split:")
    print(split_image_counts.to_string())
    print("\nClass image counts by split:")
    print(class_counts.to_string())
    print("\nLesion counts by split:")
    print(split_lesion_counts.to_string())
    print(f"\nPairwise lesion_id overlaps: {overlap_counts}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()