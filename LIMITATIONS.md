# Limitations

## Dataset provenance and watermark distribution

The manifest has no `source`, `origin`, or provenance field; its recorded fields are `md5`, `filename`, `category`, `disease`, `severity`, `split`, and `path`. Filename patterns also do not identify DermNet images. Representative images show visible DermNet watermarks across multiple disease and severity groups, so watermark presence is clearly mixed within classes. However, an exact class-level proportion cannot be computed from the current metadata. Without manually labeling all 4,377 images or adding an image-based detector, we should not claim that DermNet images are disproportionately represented in any disease or severity group. The dataset therefore lacks auditable source metadata, and the distribution and possible label correlation of watermarked images were not systematically quantified.

## Severity-label provenance

The methodology used to assign the Mild, Moderate, and Severe ground-truth labels is not documented or independently verifiable from the dataset materials available in this project. No specific clinical grading scale or expert annotation procedure is assumed here.

## Explainability and shortcut-learning risk

Grad-CAM inspection indicates that the disease classifier usually attends to lesion-bearing skin, but its attention is coarse. In some images, activation is also influenced by watermarks, image borders, hair, and surrounding background context. These cues create a potential shortcut-learning risk and mean that the heatmaps should not be interpreted as proof of clinically appropriate reasoning.

## Residual leakage risk

The original `data/dataset_manifest_v2.csv` stored hashes did not consistently match the image bytes: 1,121 values differed. Recomputing hashes from the referenced files found 421 exact-content hashes spanning multiple splits (1,060 rows affected). `data/dataset_manifest_v2_fixed.csv` retains all 5,022 rows, stores byte-verified hashes, and assigns each exact-content group to its majority split (ties go to train); verification found zero exact-content hashes across clinical splits. The merged v3 manifest carries this corrected clinical portion forward and has zero byte-hash overlap between clinical and HAM10000 images. HAM10000's lesion-grouped split also has zero lesion-ID overlap between splits. These checks do not detect near-duplicate images, transformed copies with changed bytes, or repeated clinical images of the same patient/lesion under different content; such residual leakage was not ruled out.

## Eczema ordinal-model training status

The Eczema ordinal severity model was completed through epoch 20. Validation MAE increased during epochs 16-20, so the epoch-15 checkpoint remained the best checkpoint; the earlier interruption therefore did not cost performance because the model had already converged. On the held-out test split, it outperformed the fully trained categorical baseline (69.69% versus 64.46% accuracy; ordinal MAE 0.3136 versus 0.3868).

Current `infer.py` and `app_pytorch.py` route Acne/Eczema to the requested categorical checkpoints (`runs/severity_acne/best_model.pt` and `runs/severity_eczema/best_model.pt`), not the ordinal checkpoints. The categorical heads have lower previously measured standalone accuracy than their ordinal alternatives (Acne 82.08% vs 88.53%; Eczema 64.46% vs 69.69%). The categorical severity confidence is a softmax score, not a calibrated probability. Normal, Other, NV, MEL, and BKL do not receive severity estimates.

## Out-of-distribution rejection

The original three-class Stage 1 model had no rejection class and was not trained on non-skin images. The v2 Stage 1 model added an `Other` class trained using 645 negative examples in total: 454 training, 97 validation, and 94 test images. The current v3 classifier retains `Other`, but also has NV/MEL/BKL dermoscopy classes. Neither `Other` nor any other label guarantees rejection of arbitrary out-of-distribution inputs. On the tested text-table screenshot, v2 predicted Other at 99.915%; v3 predicted Other at 99.6%. On the cardboard-box image, v2 predicted Other at 99.996% and v3 at 99.6%. A household coffee-mug image was predicted Other by v2 at 99.9997%. These are a few fixed examples, not evidence of general out-of-distribution reliability. Confidence scores are not calibrated reliability signals for arbitrary inputs.

An HSV skin-pixel threshold was also evaluated as a possible non-skin gate. Across the 663-image test set used in that analysis, skin-pixel percentages ranged from 0% to over 90%, and one genuine clinical image scored exactly 0%, the same as a text-screenshot false positive. A fixed color threshold could therefore reject real clinical images, including close-up hand/foot images and images with atypical lighting or color balance, so no such gate was deployed.

## Residual definition of the Normal class

The dataset's "Normal" class was defined as "not Acne, not Eczema," rather than as clinically verified healthy skin. It contains 16 of 639 images (2.5%) whose filenames explicitly depict perioral dermatitis, a distinct skin condition outside the project's three trained disease categories. In end-to-end pipeline validation, one such image (`Normal_perioral-dermatitis-9_0_4296.jpeg`) was misclassified as Acne and consequently triggered severity assessment. This was a downstream consequence of the upstream Stage 1 error, not a severity-gating defect. More broadly, "Normal" is a residual category of images that were neither Acne nor Eczema, rather than a clinically verified healthy-skin category; this may affect model robustness on visually similar but out-of-scope conditions.

## Ringworm generalization

Ringworm is outside the trained disease labels. On the tested image, the original three-class model predicted Normal at 96.77%; the v2 model predicted Normal at 44.32%, assigning 29.75% to Other, 25.76% to Acne, and 0.17% to Eczema. The current v3 model predicted Normal at 84.2%, Eczema at 11.6%, and Other at 0.3%. The top prediction remains incorrect and is again highly confident. This single-image probe demonstrates that adding `Other` and dermoscopy classes does not provide reliable open-set rejection; ringworm and other unseen conditions require explicit out-of-scope handling.

## Merged-domain v3 performance

The current seven-class model was evaluated on the complete 1,876-image test split in `data/dataset_manifest_v3.csv`. Overall accuracy was 88.65%, macro precision 0.8808, macro recall 0.8783, and macro F1 0.8761. Class metrics were Acne P/R/F1 0.9918/0.9308/0.9603 (130); Eczema 0.9777/0.9865/0.9821 (222); Normal 0.9406/0.9896/0.9645 (96); Other 0.9895/1.0000/0.9947 (94); NV 0.9131/0.9178/0.9154 (985); MEL 0.7099/0.5225/0.6019 (178); BKL 0.6432/0.8012/0.7135 (171).

On the 1,334 HAM10000 test images alone, accuracy was 85.01% and macro F1 0.7436. Melanoma recall was 52.25%; 60/178 MEL images were classified as NV and 25/178 as BKL. BKL precision was 64.32%. This error profile matters because the dermoscopy classes have uneven supports and errors are concentrated among visually similar lesion classes. The test set is now a fixed reference and must not be used for model selection, threshold tuning, or repeated experimentation. The three-image-per-class sanity sample is too small to estimate performance. Clinical photos and dermoscopic images also come from different image domains; strong aggregate scores do not establish equivalent behavior across clinical settings or acquisition devices.

