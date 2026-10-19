# Project Setup Guide

This guide reflects the current repository state as implemented in the checked-in scripts, manifests, and configuration files. It is intentionally limited to commands and layout assumptions that are actually used by the project.

## 1. Python and editor environment

The project is designed to run from a local Python virtual environment in the repository root.

Current repo layout includes:

- `.venv/` - the active project virtual environment
- `requirements.txt` - pinned dependency list
- project scripts in the repository root such as `infer.py`, `train_stage1_v2.py`, `train_stage2.py`, `train_severity_ordinal_pytorch.py`, and `train_ham10000_resnet50.py`

The repository's active environment has been created as `.venv` and is using Python 3.10.

For VS Code, use the project venv as the selected Python interpreter:

- Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Then verify the interpreter and run commands with:

```powershell
.\.venv\Scripts\python.exe --version
```

## 2. Dependency installation

The dependency list is defined in `requirements.txt`:

```text
torch==2.6.0
torchvision==0.21.0
pandas
numpy
pillow
scikit-learn
matplotlib
seaborn
gradio
```

Install project dependencies from the repo root using the project venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This is the install path the project currently expects. `gradio` is included in the dependency list and is required by `app_pytorch.py`.

## 3. Required repository structure

The project root currently contains the following relevant files and folders:

```text
skin_disease_project/
├── .gitignore
├── .venv/
├── app_pytorch.py
├── data/
│   ├── clean_raw/
│   ├── splits/
│   ├── dataset_manifest.csv
│   ├── dataset_manifest_v2.csv
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   ├── ham10000_3class/
│   └── ...
├── infer.py
├── requirements.txt
├── runs/
│   ├── efficientnet_b0/
│   ├── resnet50/
│   ├── resnet50_v2/
│   ├── severity_acne_ordinal/
│   ├── severity_eczema_ordinal/
│   └── ...
├── train_stage1.py
├── train_stage1_v2.py
├── train_stage2.py
├── train_severity_ordinal_pytorch.py
├── train_ham10000_resnet50.py
├── prepare_dataset.py
├── prepare_ham10000_experiment.py
├── README.txt
├── PROJECT_HANDOFF.md
├── LIMITATIONS.md
├── run_stage1.bat
└── ...
```

## 4. Required dataset layout

### Obtaining and placing data

Raw image datasets and newly generated manifests are intentionally not stored
in this repository because the source and redistribution terms could not be
verified for every input. Obtain each dataset and any permitted metadata from
its original source under that source's terms; this project does not grant
redistribution rights. Create or obtain the required local manifests before
running the affected experiments.

For the clinical severity and disease-classification experiments, provide the
permitted local copy of `data/dataset_manifest_v3.csv` and place the image files
at its referenced paths under `data/` (the manifest paths use backslashes).
Keep the manifest and image directory structure unchanged, or update the local
manifest paths to match. The training commands use
`data/dataset_manifest_v3.csv` and `data` as the image root by default. The
manifest is excluded from this repository until its redistribution terms are
confirmed.

For the HAM10000 experiments, obtain the HAM10000 images and metadata from the
original dataset source. Place extracted images in
`data/HAM10000_images_part_1/` and `data/HAM10000_images_part_2/`, and metadata
at `data/HAM10000_metadata.csv`. Provide or generate the local
`data/ham10000_3class/split_manifest.csv`; it refers to those image paths.
The source images, metadata, and derived manifest remain local and are
excluded from Git until redistribution terms are confirmed.

The current clinical-image training and inference pipeline relies on the manifest and image structure expected by the training scripts and `infer.py`.

### Clinical dataset format

The required local manifest files such as `data/dataset_manifest.csv`,
`data/dataset_manifest_v2.csv`, and `data/dataset_manifest_v3.csv` contain rows
with fields including:

- `md5`
- `filename`
- `category`
- `disease`
- `severity`
- `split`
- `path`

The `path` values in the manifest use a repo-relative path such as:

```text
splits\train\Acne_Mild\Acne_Mild_acne-closed-comedo-15.jpg
```

The scripts resolve image paths by checking the manifest path first, then `clean_raw/<category>/<filename>`, and then recursive search under the data root. This means the project expects the image files to be present under the data tree, with the expected split and category folders present.

The training script `train_stage1_v2.py` defines the class order as:

```python
CLASS_NAMES = ["Acne", "Eczema", "Normal", "Other"]
```

and expects images to be accessible in the project data tree.

### Stage 1 image directory expected by the scripts

The repo's earlier stage-1 setup and `run_stage1.bat` are built around:

```text
data/
└── clean_raw/
    ├── Acne/
    ├── Eczema/
    ├── Normal/
    └── ...
```

The script `train_stage1.py` explicitly supports the `clean_raw/<disease>/<filename>` layout and also falls back to recursive lookup.

### HAM10000 manifest layout

The separate HAM10000 pipeline expects the local
`data/ham10000_3class/split_manifest.csv` with required columns:

```text
dx, lesion_id, path, split
```

The script `train_ham10000_resnet50.py` checks that:

- all split entries are in `dx` values `nv`, `mel`, `bkl`
- `path` values actually exist
- no `lesion_id` overlaps exist between train/val/test splits

This is the leak-safe grouped split that the current HAM10000 baseline expects.

## 5. Required trained checkpoints and exact locations

The current clinical inference pipeline uses the checkpoint paths embedded in `infer.py` and `app_pytorch.py`.

Default inference checkpoints in `infer.py`:

- Stage 1 disease model:
  - `runs/resnet50_v2/best_model.pt`
- Acne severity model:
  - `runs/severity_acne_ordinal/best_model.pt`
- Eczema severity model:
  - `runs/severity_eczema_ordinal/best_model.pt`

These files are required to exist before running inference.

The project currently includes the corresponding run directories under `runs/` and they contain the expected `best_model.pt` files. For example:

```text
runs/
├── resnet50_v2/
│   └── best_model.pt
├── severity_acne_ordinal/
│   └── best_model.pt
├── severity_eczema_ordinal/
│   └── best_model.pt
└── ...
```

If any of these files are missing, `infer.py` raises `FileNotFoundError` during model loading.

## 6. Exact inference commands

From the project root, with the venv activated:

```powershell
.\.venv\Scripts\python.exe infer.py "C:\path\to\image.jpg"
```

or, in a shell where the venv is already active:

```powershell
python infer.py "C:\path\to\image.jpg"
```

### JSON output

```powershell
.\.venv\Scripts\python.exe infer.py "C:\path\to\image.jpg" --json
```

### Demo mode

The script includes an automated demo mode that loads the held-out manifest and walks a few representative test samples:

```powershell
.\.venv\Scripts\python.exe infer.py --demo
```

This is the exact CLI behavior implemented in the script.

### Notes on checkpoint overrides

The CLI also supports overriding checkpoint paths explicitly:

```powershell
.\.venv\Scripts\python.exe infer.py "C:\path\to\image.jpg" --stage1-ckpt "runs\resnet50_v2\best_model.pt" --acne-ckpt "runs\severity_acne_ordinal\best_model.pt" --eczema-ckpt "runs\severity_eczema_ordinal\best_model.pt"
```

These are valid options defined in `infer.py`.

## 7. Expected inference output

`infer.py` prints a CLI summary with:

- image file name
- device (`CUDA` or `CPU`)
- Stage 1 disease prediction
- disease confidence and per-class probabilities
- Stage 2 severity level for Acne/Eczema
- `N/A` severity for `Normal` and `Other`
- a clinical summary string

The output is generated by `format_cli_output(result)` with sections titled:

- `STAGE 1: DISEASE IDENTIFICATION`
- `STAGE 2: SEVERITY ASSESSMENT`
- `CLINICAL SUMMARY`

The model classes are exactly:

```python
STAGE1_CLASSES = ["Acne", "Eczema", "Normal", "Other"]
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
```

The script returns a dictionary containing keys such as:

- `image_path`
- `disease`
- `disease_confidence`
- `disease_probabilities`
- `severity`
- `severity_confidence`
- `severity_details`
- `clinical_summary`

For `Normal` and `Other`, the severity output is deliberately `N/A`.

## 8. Gradio app

The project includes `app_pytorch.py`, which launches a Gradio interface and loads the same Stage 1/Stage 2 checkpoints.

The command to run it is:

```powershell
.\.venv\Scripts\python.exe app_pytorch.py
```

This app expects `gradio` to be installed in the environment, and `requirements.txt` includes it.

## 9. Common troubleshooting

### Missing checkpoint

Symptoms:

- `FileNotFoundError: Model checkpoint not found: ...`

Cause:

- The expected file is not present under the path used by `infer.py`.

Check:

```powershell
dir runs\resnet50_v2
dir runs\severity_acne_ordinal
dir runs\severity_eczema_ordinal
```

### Image path does not exist

Symptoms:

- `FileNotFoundError: Image does not exist: ...`

Cause:

- The image path passed to `infer.py` is incorrect or the file is not present.

Check:

```powershell
Test-Path "C:\path\to\image.jpg"
```

### Unsupported architecture error

Symptoms:

- `ValueError: Unsupported architecture: ...`

Cause:

- The checkpoint `model_name` or `arch` metadata does not match a supported `resnet50` or `efficientnet_b0` architecture.

This project's loaders support `resnet50` and `efficientnet_b0` only for the currently used model checkpoint structure.

### Gradio import error

Symptoms:

- `ModuleNotFoundError: No module named 'gradio'`

Cause:

- The environment does not have `gradio` installed.

Fix:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

or install just the missing dependency if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install gradio
```

### Mixed path format issues on Windows

The repo contains manifest entries using Windows-style paths such as:

```text
splits\train\Acne_Mild\Acne_Mild_acne-closed-comedo-15.jpg
```

Use path strings that resolve correctly in the local environment; the scripts do resolve manifest paths and fallback paths but do expect them to be valid on the machine where the project is run.

## 10. Practical setup checklist

Use this checklist before running inference:

1. Open VS Code in the repo root.
2. Select the project `.venv` interpreter.
3. Install dependencies from `requirements.txt`.
4. Confirm the project data tree exists and the manifests are present.
5. Confirm the checkpoint files exist:
   - `runs/resnet50_v2/best_model.pt`
   - `runs/severity_acne_ordinal/best_model.pt`
   - `runs/severity_eczema_ordinal/best_model.pt`
6. Run a single image through `infer.py`.
7. If needed, use `--json` for machine-readable output.
8. If you want the GUI, launch `app_pytorch.py`.

## 11. Notes from the repo

The repository includes older and more experimental files such as `train_ham10000_resnet50.py`, `prepare_ham10000_experiment.py`, and HAM10000 split artifacts under `data/ham10000_3class/` and `runs/ham10000_*`. Those are separate from the main clinical pipeline and are not required for the default `infer.py` command path unless you are explicitly running the HAM10000 experiment.

For the default clinical pipeline described here, the required runtime path is the Stage 1 + Stage 2 checkpoint set under `runs/` plus the `data/` manifest and image tree.
