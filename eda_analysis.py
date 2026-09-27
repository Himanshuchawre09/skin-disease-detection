import os
import pandas as pd
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

MANIFEST_PATH = r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data\dataset_manifest.csv"
DATA_DIR = r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\data"
REPORT_PATH = r"C:\Users\asus\.gemini\antigravity\scratch\skin_disease_project\eda_report.txt"

def run_eda():
    print("=" * 60)
    print("PHASE 5: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} images.")

    # 1. Inspect image dimensions and color modes
    widths = []
    heights = []
    aspect_ratios = []
    modes = defaultdict(int)

    print("Auditing image resolutions and channels...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        full_path = os.path.join(DATA_DIR, row["path"])
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(round(w / h, 2))
                modes[img.mode] += 1
        except Exception as e:
            print(f"Error reading {full_path}: {e}")

    df["width"] = widths
    df["height"] = heights
    df["aspect_ratio"] = aspect_ratios

    # Summary statistics
    w_min, w_max, w_mean, w_median = min(widths), max(widths), sum(widths)/len(widths), sorted(widths)[len(widths)//2]
    h_min, h_max, h_mean, h_median = min(heights), max(heights), sum(heights)/len(heights), sorted(heights)[len(heights)//2]

    report = []
    report.append("=" * 65)
    report.append("DATASET EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    report.append("=" * 65)
    report.append(f"Total Authentic Images Analyzed: {len(df)}")
    report.append("\n[1. COLOR CHANNELS / COLOR SPACE]")
    for mode, count in modes.items():
        report.append(f"  Mode {mode:6s}: {count:5d} images ({count/len(df)*100:.1f}%)")

    report.append("\n[2. RESOLUTION METRICS]")
    report.append(f"  Width:  Min = {w_min}px, Max = {w_max}px, Mean = {w_mean:.1f}px, Median = {w_median}px")
    report.append(f"  Height: Min = {h_min}px, Max = {h_max}px, Mean = {h_mean:.1f}px, Median = {h_median}px")
    report.append(f"  Recommended Model Input Size: 224x224 (Standard for EfficientNet/ResNet)")

    report.append("\n[3. DISEASE DISTRIBUTION (STAGE 1)]")
    disease_counts = df.groupby(["disease", "split"]).size().unstack(fill_value=0)
    disease_counts["Total"] = disease_counts.sum(axis=1)
    disease_counts["Percentage"] = (disease_counts["Total"] / len(df) * 100).round(1).astype(str) + "%"
    report.append(disease_counts.to_string())

    report.append("\n[4. SEVERITY DISTRIBUTION (STAGE 2 - ACNE)]")
    acne_df = df[df["disease"] == "Acne"]
    acne_sev = acne_df.groupby(["severity", "split"]).size().unstack(fill_value=0)
    acne_sev["Total"] = acne_sev.sum(axis=1)
    report.append(acne_sev.to_string())

    report.append("\n[5. SEVERITY DISTRIBUTION (STAGE 2 - ECZEMA)]")
    eczema_df = df[df["disease"] == "Eczema"]
    eczema_sev = eczema_df.groupby(["severity", "split"]).size().unstack(fill_value=0)
    eczema_sev["Total"] = eczema_sev.sum(axis=1)
    report.append(eczema_sev.to_string())

    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nReport written to: {REPORT_PATH}")

if __name__ == "__main__":
    run_eda()
