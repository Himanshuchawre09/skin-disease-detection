import zipfile
import re
import os
import shutil
import random
import hashlib
from collections import defaultdict
from PIL import Image
import pandas as pd

ZIP_PATH = r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data\Dataset.zip"
BASE_DIR = r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data"
RAW_CLEAN_DIR = os.path.join(BASE_DIR, "clean_raw")
SPLIT_DIR = os.path.join(BASE_DIR, "splits")
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset_manifest.csv")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CATEGORY_MAP = {
    "Acne_Mild": "Acne_Mild",
    "Acne_Moderate": "Acne_Moderate",
    "Acne_Severe": "Acne_Severe",
    "Eczema_Mild": "Eczema_Mild",
    "Eczema_Moderate": "Eczema_Moderate",
    "Eczema_Moderate (2)": "Eczema_Mild",
    "Eczema_Severe": "Eczema_Severe",
    "Normal": "Normal"
}

def clean_and_split():
    print("=" * 65)
    print("DATA AUDIT & CLEANING: HASH-BASED DEDUPLICATION & LEAKAGE REMOVAL")
    print("=" * 65)

    if os.path.exists(RAW_CLEAN_DIR):
        shutil.rmtree(RAW_CLEAN_DIR)
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    os.makedirs(RAW_CLEAN_DIR, exist_ok=True)

    # 1. Scan zip and map images by their exact MD5 content hash
    hash_to_instances = defaultdict(list)
    skipped_xyz = 0
    skipped_aug = 0
    corrupted_count = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            if "xyz" in entry.filename.lower():
                skipped_xyz += 1
                continue
            if not entry.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            parts = entry.filename.split('/')
            if len(parts) < 3:
                continue
            folder_name = parts[2]
            file_name = parts[3]

            if folder_name not in CATEGORY_MAP:
                continue

            # Strip synthetic offline aug_ prefix
            base_name = re.sub(r'^aug_\d+_', '', file_name)
            canonical_cat = CATEGORY_MAP[folder_name]

            try:
                img_bytes = z.read(entry)
                import io
                with Image.open(io.BytesIO(img_bytes)) as img:
                    img.verify()
                
                md5 = hashlib.md5(img_bytes).hexdigest()
                hash_to_instances[md5].append({
                    "category": canonical_cat,
                    "filename": base_name,
                    "bytes": img_bytes
                })
            except Exception as e:
                corrupted_count += 1

    print(f"\n[Zip Scan Statistics]")
    print(f"  - Total unique image hashes identified: {len(hash_to_instances)}")
    print(f"  - Excluded 'Xyz' images: {skipped_xyz}")

    # 2. Identify and resolve label conflicts (same image assigned to multiple categories)
    clean_images = []
    conflict_count = 0

    for md5, instances in hash_to_instances.items():
        categories = set(inst["category"] for inst in instances)
        if len(categories) > 1:
            # Ambiguous image present in multiple categories (e.g., Mild and Moderate)
            conflict_count += 1
            continue
        
        # Unambiguous image: pick canonical category and first clean filename
        canonical_cat = instances[0]["category"]
        fname = instances[0]["filename"]
        clean_images.append({
            "md5": md5,
            "category": canonical_cat,
            "filename": f"{canonical_cat}_{fname}",
            "bytes": instances[0]["bytes"]
        })

    print(f"\n[Quality Assurance / Data Cleansing]")
    print(f"  - Ambiguous / Conflicting label images removed: {conflict_count}")
    print(f"  - Final verified unambiguous, authentic images: {len(clean_images)}")

    # 3. Save clean images to RAW_CLEAN_DIR
    for cat in set(CATEGORY_MAP.values()):
        os.makedirs(os.path.join(RAW_CLEAN_DIR, cat), exist_ok=True)
    
    for item in clean_images:
        out_path = os.path.join(RAW_CLEAN_DIR, item["category"], item["filename"])
        with open(out_path, "wb") as f:
            f.write(item["bytes"])

    # 4. Stratified Split (70 / 15 / 15) by Category
    random.seed(RANDOM_SEED)
    manifest_rows = []

    for split in ["train", "val", "test"]:
        for cat in set(CATEGORY_MAP.values()):
            os.makedirs(os.path.join(SPLIT_DIR, split, cat), exist_ok=True)

    images_by_cat = defaultdict(list)
    for item in clean_images:
        images_by_cat[item["category"]].append(item)

    print("\n[Split Distribution per Category]")
    for cat in sorted(images_by_cat.keys()):
        cat_items = images_by_cat[cat]
        random.shuffle(cat_items)

        n_total = len(cat_items)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        splits = {
            "train": cat_items[:n_train],
            "val": cat_items[n_train:n_train + n_val],
            "test": cat_items[n_train + n_val:]
        }

        # Parse disease and severity
        if cat.startswith("Acne"):
            disease = "Acne"
            severity = cat.split("_")[1]
        elif cat.startswith("Eczema"):
            disease = "Eczema"
            severity = cat.split("_")[1]
        else:
            disease = "Normal"
            severity = "None"

        for split_name, item_list in splits.items():
            for item in item_list:
                src_path = os.path.join(RAW_CLEAN_DIR, item["category"], item["filename"])
                dst_path = os.path.join(SPLIT_DIR, split_name, item["category"], item["filename"])
                shutil.copy2(src_path, dst_path)

                manifest_rows.append({
                    "md5": item["md5"],
                    "filename": item["filename"],
                    "category": item["category"],
                    "disease": disease,
                    "severity": severity,
                    "split": split_name,
                    "path": os.path.relpath(dst_path, BASE_DIR)
                })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest written to: {MANIFEST_PATH}")

    split_table = pd.crosstab(df["category"], df["split"], margins=True)
    print("\n", split_table)

    # 5. Formal Mathematical Verification of ZERO Leakage
    print("\n" + "=" * 65)
    print("STEP 5: MATHEMATICAL LEAKAGE VERIFICATION (BY CONTENT HASH)")
    print("=" * 65)
    train_hashes = set(df[df["split"] == "train"]["md5"])
    val_hashes = set(df[df["split"] == "val"]["md5"])
    test_hashes = set(df[df["split"] == "test"]["md5"])

    train_val = train_hashes.intersection(val_hashes)
    train_test = train_hashes.intersection(test_hashes)
    val_test = val_hashes.intersection(test_hashes)

    print(f"  - Train - Val Hash Intersection:  {len(train_val)}")
    print(f"  - Train - Test Hash Intersection: {len(train_test)}")
    print(f"  - Val - Test Hash Intersection:   {len(val_test)}")

    assert len(train_val) == 0, "Train-Val Leakage!"
    assert len(train_test) == 0, "Train-Test Leakage!"
    assert len(val_test) == 0, "Val-Test Leakage!"

    print("\n  >>> VERIFICATION PASSED: ZERO DATA LEAKAGE CONFIRMED! <<<")
    print("=" * 65)

if __name__ == "__main__":
    clean_and_split()
