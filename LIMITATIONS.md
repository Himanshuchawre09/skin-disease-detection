# Limitations

## Dataset provenance and watermark distribution

The manifest has no `source`, `origin`, or provenance field; its recorded fields are `md5`, `filename`, `category`, `disease`, `severity`, `split`, and `path`. Filename patterns also do not identify DermNet images. Representative images show visible DermNet watermarks across multiple disease and severity groups, so watermark presence is clearly mixed within classes. However, an exact class-level proportion cannot be computed from the current metadata. Without manually labeling all 4,377 images or adding an image-based detector, we should not claim that DermNet images are disproportionately represented in any disease or severity group. The dataset therefore lacks auditable source metadata, and the distribution and possible label correlation of watermarked images were not systematically quantified.

## Severity-label provenance

The methodology used to assign the Mild, Moderate, and Severe ground-truth labels is not documented or independently verifiable from the dataset materials available in this project. No specific clinical grading scale or expert annotation procedure is assumed here.

## Explainability and shortcut-learning risk

Grad-CAM inspection indicates that the disease classifier usually attends to lesion-bearing skin, but its attention is coarse. In some images, activation is also influenced by watermarks, image borders, hair, and surrounding background context. These cues create a potential shortcut-learning risk and mean that the heatmaps should not be interpreted as proof of clinically appropriate reasoning.

## Residual leakage risk

Content-hash deduplication removes exact duplicate files across splits, but it does not detect near-duplicate images of the same lesion taken from different angles or reused with different crops or transformations. A residual risk of same-lesion leakage between splits therefore remains and was not fully ruled out.

## Eczema ordinal-model training status

The Eczema ordinal severity model was completed through epoch 20. Validation MAE increased during epochs 16-20, so the epoch-15 checkpoint remained the best checkpoint; the earlier interruption therefore did not cost performance because the model had already converged. On the held-out test split, it outperformed the fully trained categorical baseline (69.69% versus 64.46% accuracy; ordinal MAE 0.3136 versus 0.3868).

## Out-of-distribution rejection

The original three-class Stage 1 model had no rejection class and was not trained on non-skin images. The v2 Stage 1 model adds an `Other` class trained using 645 negative examples in total: 454 training, 97 validation, and 94 test images. This is a learned non-skin class, not a guarantee that arbitrary out-of-distribution inputs will be rejected. On the tested text-table screenshot, the original model predicted Eczema at 94.41% confidence, while v2 predicted Other at 99.915%. On the cardboard-box image, v2 predicted Other at 99.996%; a household coffee-mug image was also predicted Other at 99.9997%. These examples show improved behavior on the tested negatives, but do not establish general out-of-distribution reliability. Confidence scores should not be treated as calibrated reliability signals on arbitrary inputs.

An HSV skin-pixel threshold was also evaluated as a possible non-skin gate. Across the 663-image test set used in that analysis, skin-pixel percentages ranged from 0% to over 90%, and one genuine clinical image scored exactly 0%, the same as a text-screenshot false positive. A fixed color threshold could therefore reject real clinical images, including close-up hand/foot images and images with atypical lighting or color balance, so no such gate was deployed.

## Residual definition of the Normal class

The dataset's "Normal" class was defined as "not Acne, not Eczema," rather than as clinically verified healthy skin. It contains 16 of 639 images (2.5%) whose filenames explicitly depict perioral dermatitis, a distinct skin condition outside the project's three trained disease categories. In end-to-end pipeline validation, one such image (`Normal_perioral-dermatitis-9_0_4296.jpeg`) was misclassified as Acne and consequently triggered severity assessment. This was a downstream consequence of the upstream Stage 1 error, not a severity-gating defect. More broadly, "Normal" is a residual category of images that were neither Acne nor Eczema, rather than a clinically verified healthy-skin category; this may affect model robustness on visually similar but out-of-scope conditions.

## Ringworm generalization

Ringworm is outside the model's trained disease labels. On one tested ringworm image, the original three-class model predicted Normal at 96.77%. The four-class v2 model still predicted Normal, but with lower confidence (44.32%) and assigned 29.75% to Other (25.76% to Acne and 0.17% to Eczema). The probability distribution is less confidently wrong and gives appreciable mass to Other, but the top prediction remains incorrect; this is a partial improvement in uncertainty behavior, not successful ringworm recognition.

