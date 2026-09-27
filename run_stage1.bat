@echo off
set PYTHON=.venv\Scripts\python.exe
set MANIFEST=data\dataset_manifest.csv
set IMAGE_ROOT=data\clean_raw

echo ==============================
echo Training EfficientNetB0
echo ==============================
%PYTHON% train_stage1.py --manifest "%MANIFEST%" --image-root "%IMAGE_ROOT%" --model efficientnet_b0 --epochs 20 --batch-size 16 --patience 5

echo.
echo ==============================
echo Training ResNet50
echo ==============================
%PYTHON% train_stage1.py --manifest "%MANIFEST%" --image-root "%IMAGE_ROOT%" --model resnet50 --epochs 20 --batch-size 16 --patience 5

echo.
echo Training complete. Check the runs\ folder.
pause
