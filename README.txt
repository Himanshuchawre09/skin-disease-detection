
# Stage 1: Acne vs Eczema vs Normal

This package continues the project from the verified EDA stage.

## Expected project layout

skin_disease_project/
├── .venv/
├── data/
│   ├── clean_raw/
│   │   ├── Acne/
│   │   ├── Eczema/
│   │   └── Normal/
│   └── dataset_manifest.csv
├── train_stage1.py
├── run_stage1.bat
└── runs/

The manifest must contain:
- filename
- disease
- split

Accepted disease values:
- Acne
- Eczema
- Normal

The script supports the clean_raw/<disease>/<filename> structure and also searches recursively if the folder layout differs.

## First run

From the project root:

python train_stage1.py --manifest data\dataset_manifest.csv --image-root data\clean_raw --model efficientnet_b0 --epochs 20 --batch-size 16 --patience 5

Then:

python train_stage1.py --manifest data\dataset_manifest.csv --image-root data\clean_raw --model resnet50 --epochs 20 --batch-size 16 --patience 5

Or simply run:

run_stage1.bat

## What this does

- Converts all images to RGB.
- Applies mild medical-image-friendly augmentation only to training data.
- Resizes images to 224x224.
- Uses ImageNet-pretrained EfficientNetB0 or ResNet50.
- Replaces the classification head with 3 classes.
- Computes class weights from the training split.
- Uses weighted CrossEntropyLoss to account for the smaller Normal class.
- Uses AdamW + ReduceLROnPlateau.
- Uses mixed precision automatically on CUDA.
- Uses early stopping based on validation loss.
- Saves only the best validation-loss checkpoint.
- Evaluates the untouched test split once at the end.
- Produces Accuracy, macro Precision, macro Recall, macro F1, per-class report and confusion matrix.
- Saves training curves.

## Important

Do not use the test set while tuning epochs, augmentations, learning rate or model choice.

The test set should remain a final evaluation set.

The two models should be compared using the same train/validation/test splits.

Do not choose the model based on accuracy alone. Inspect macro F1 and the per-class recall, especially for Normal.

## Next stage after Stage 1

After both models are trained and evaluated:

1. Compare EfficientNetB0 vs ResNet50.
2. Inspect the confusion matrices.
3. Check whether training and validation curves show overfitting.
4. Test on completely unseen external images where appropriate.
5. Select the model based on the full evidence.
6. Then build Stage 2 severity models for diseases for which the severity labels are supported by the dataset.
