# TASK: Architecture-modification comparison for the severity models

Read this whole file before writing any code. Work in the stages below and stop after each stage to report real output. Do not skip ahead.

## Goal

My teacher asked me to take several "-Net" CNN architectures, modify each by removing a layer/block and by adding a layer/block, train them all on the same task, and compare everything side by side. I will design our own final architecture later, after the results are in. That is NOT part of this task.

## Project facts (do not re-derive, do not change)

- PyTorch + torchvision, venv at `.venv\Scripts\python.exe`, Windows, RTX 4050 6GB.
- Manifest: `data/dataset_manifest_v3.csv`, columns `md5, filename, category, disease, severity, split, path`. Images are under `data/`; the `path` column uses backslashes. The clinical rows are the corrected, leak-free splits. Do NOT modify any manifest.
- Task: ordinal severity regression. Labels Mild=0, Moderate=1, Severe=2. Single scalar output `Linear(features, 1)`, MSE loss, at inference round and clip to {0,1,2}. Only rows where `disease == "Eczema"` (first) or `"Acne"` have severity.
- Preprocessing: 224x224, ImageNet mean/std. Train augmentation: RandomResizedCrop(scale 0.8-1.0), horizontal flip, rotation +-10, conservative ColorJitter. Eval: Resize((224,224)) + normalize.
- Reuse the dataset class, transforms, and metric code from `train_severity_ordinal_pytorch.py` instead of rewriting them. Read that file first.

## Hard rules

1. Do NOT touch, overwrite, or delete any existing `.pt` checkpoint, `infer.py`, `app_pytorch.py`, any manifest, or any existing `runs/` folder.
2. All new outputs go under `runs/arch_comparison/<disease>/<run_name>/`.
3. Do NOT run the full training yourself. Long runs get killed when run through the agent. You may run a 1-epoch smoke test and quick shape checks. I will run the full job in my own terminal.
4. Never invent or estimate numbers. If something was not run, say so.
5. Channel counts below are from memory. Verify every one with a dummy forward pass and use the measured value.

## Networks and edits (7 networks x 3 versions = 21 runs per disease)

Each network gets `baseline` (unmodified body, new 1-output head), `remove` (a layer or block taken out), and `add` (a layer or block added). Every version ends in AdaptiveAvgPool2d(1) -> Flatten -> Linear(measured_channels, 1), except where noted.

| Network | torchvision builder | remove | add |
|---|---|---|---|
| ResNet50 | `resnet50` | drop `layer4` (use layer1-3 output) | Squeeze-and-Excitation block after `layer4` |
| DenseNet121 | `densenet121` | drop `denseblock4` and the final `norm5` | extra Conv-BN-ReLU on the final feature map |
| EfficientNet-B0 | `efficientnet_b0` | drop the last 1x1 conv in `features` | extra Conv-BN-ReLU |
| MobileNetV2 | `mobilenet_v2` | drop the last 1x1 conv in `features` | extra Conv-BN-ReLU |
| SqueezeNet1_1 | `squeezenet1_1` | drop the last Fire module | extra Conv-BN-ReLU before pooling |
| ShuffleNetV2 | `shufflenet_v2_x1_0` | drop the final stage (stage4) | extra Conv-BN-ReLU |
| GoogLeNet | `googlenet` | drop `inception5b` | extra Conv-BN-ReLU after `inception5b` |

Notes:
- GoogLeNet: load with `aux_logits=True` for the pretrained weights, then set `aux_logits=False` and remove `aux1`/`aux2` so the forward pass returns a tensor in train mode. Confirm `transform_input` works with our ImageNet-normalized input.
- SqueezeNet's classifier is convolutional, so build the same pooled `Linear(channels, 1)` head as the others instead of using its native classifier.
- ShuffleNetV2: inspect `print(model)` and pick the correct attribute names for the stage and final conv. If "final stage" is ambiguous, tell me what you chose and why.
- The added layer for each network is randomly initialized; the layers you kept keep pretrained weights. Use ImageNet pretrained weights for every network (check that the weights are cached or downloadable; say if internet is needed on first run).
- Put all model construction in one file `arch_variants.py` with a single function `build_model(network, variant)` that returns the model and a small metadata dict (feature channels, one-line description of the edit).

## STAGE A: build and verify (no training)

1. Write `arch_variants.py`.
2. For all 21 combinations, run a dummy forward pass on a `(2, 3, 224, 224)` tensor in both train and eval mode. Assert the output shape is `(2, 1)`.
3. Print and save `runs/arch_comparison/architecture_summary.csv` with: network, variant, feature channels, parameter count (total and trainable), and the edit description.
4. Report any combination that failed and how you fixed it. STOP here and show me the table.

## STAGE B: training script and smoke test

1. Write `train_arch_comparison.py` with arguments: `--disease` (Eczema or Acne), `--networks` (default all), `--variants` (default all three), `--epochs` (default 20), `--batch-size` (default 16), `--lr` (default 3e-4), `--patience` (default 5), `--seed` (default 42), `--num-workers` (default 2), `--device auto`.
2. Identical settings for every run: same seed, same split, same augmentation, AdamW or whatever `train_severity_ordinal_pytorch.py` uses (match it), same scheduler, MSE loss.
3. Select the best epoch on validation MAE. Evaluate the test split once, after training.
4. For each run, save to its folder: `best_model.pt`, `history.json` (per-epoch train loss, val loss, val MAE, seconds per epoch), `metrics.json` (test accuracy after rounding, test MAE, per-class precision/recall/F1, confusion matrix, parameter count, average training seconds per epoch, inference milliseconds per image measured on the test set with the model in eval mode and no_grad), and `predictions.csv`.
5. Make it resumable: if `metrics.json` already exists for a run, skip it. An interrupted job must be restartable with the same command without losing finished runs.
6. Print a one-line summary after each run, and handle one run failing without killing the whole job (log the error, continue).
7. Run a smoke test only: `--networks resnet50 --variants baseline --epochs 1`, writing to a temporary folder you delete afterward. Show me the output. STOP here.

## STAGE C: command for me

Give me the exact command to run the full Eczema job myself, including a note on estimated time based on the 1-epoch smoke test timing (state clearly that it is an estimate). Also tell me how to run a single network or the Acne job separately.

## STAGE D (only after I report that training finished): comparison plots

Write `plot_comparison.py` that reads every `history.json` and `metrics.json` under `runs/arch_comparison/<disease>/` and saves to `runs/arch_comparison/<disease>/comparison/`:

1. Training loss curves, all runs on one figure.
2. Validation MAE curves, all runs on one figure.
3. Bar charts for: test accuracy, test MAE, parameter count, inference ms per image, training seconds per epoch.
4. A grid of confusion matrices, one per run.
5. `master_table.csv` and `master_table.md` with every number, sorted by test MAE.

Plot conventions: one fixed color per network, line style per variant (baseline solid, remove dashed, add dotted). Label axes and units. Keep figures readable for 21 runs, splitting into per-network panels if one figure gets too crowded.

## Reporting

- Report real numbers only. Note that the test sets are small (about 130 Acne and 222 Eczema images), so differences of one or two points are within noise. State this in the summary.
- If a variant is worse than its baseline, say so plainly. That is a valid result.
- At the end, list every file you created or changed.
