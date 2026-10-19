# Skin Disease Detection and Severity Assessment

PyTorch and torchvision project for image-based skin-disease classification and Acne/Eczema severity assessment. The repository includes training and inference code, experiment summaries, and selected aggregate plots. Raw image datasets are intentionally not included.

## Models and inference

The seven-class Stage 1 classifier recognizes Acne, Eczema, Normal, Other, NV, MEL, and BKL. Stage 2 estimates ordinal severity for Acne and Eczema only. The default inference checkpoints are stored under `runs/` and are tracked with Git LFS.

Clone with Git LFS installed, then retrieve the checkpoint objects:

```powershell
git lfs install
git lfs pull
```

Install the Python dependencies from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run inference on a local image:

```powershell
.\.venv\Scripts\python.exe infer.py "C:\path\to\image.jpg"
```

Launch the Gradio demo:

```powershell
.\.venv\Scripts\python.exe app_pytorch.py
```

## Data

Image datasets and newly generated manifests are excluded because source and redistribution terms could not be verified for every input. Obtain the data and its permitted metadata from the original providers, then place local manifests and images according to [SETUP.md](SETUP.md). The manifests are required by the training commands but are not included in this publication.

The clinical experiments expect `data/dataset_manifest_v3.csv`. HAM10000 experiments expect the lesion-grouped `data/ham10000_3class/split_manifest.csv`, with source images and metadata kept local. Do not merge the clinical and dermoscopy image domains without a separately defined experiment.

## Training and comparisons

The repository contains the clinical training scripts, HAM10000 experiment scripts, and the architecture comparison implementation (`arch_variants.py`, `train_arch_comparison.py`, and `plot_comparison.py`). Training requires the local datasets described in [SETUP.md](SETUP.md). The checked-in metrics and tables are experiment outputs; review [LIMITATIONS.md](LIMITATIONS.md) before interpreting them.

The `report/tables/` directory contains machine-readable aggregate results. `report/figures/` includes selected aggregate plots and diagrams. Per-image predictions, raw images, Grad-CAM examples containing source photos, and experiment checkpoints outside the default inference set are not included.

## Reproducibility and limitations

Python dependencies are listed in `requirements.txt`; setup details and expected data paths are in [SETUP.md](SETUP.md). Dataset provenance and label-methodology uncertainties, shortcut-learning risks, and held-out test limitations are documented in [LIMITATIONS.md](LIMITATIONS.md).
