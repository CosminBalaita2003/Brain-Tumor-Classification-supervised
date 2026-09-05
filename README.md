# Brain Tumor Classification (Supervised)

This project performs binary MRI classification: `tumor` versus `not_tumor`.
Glioma, meningioma, and pituitary images map to `tumor`; no-tumor images map to
`not_tumor`. It compares Random Forest, LightGBM, and a Logistic Regression baseline using handcrafted
intensity/texture, HOG, or multi-scale wavelet features.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original data is stored under `archive/Training/<class>/` and
`archive/Testing/<class>/`. Create fresh train/validation/test manifests with:

```bash
python prepare_dataset.py
```

The script deduplicates each original partition and splits `archive/Training`
into seeded, stratified 80/20 train/validation sets. `archive/Testing` remains the
independent test set. If image content occurs in both original partitions, it is
kept only in test. The script writes `train.csv`, `val.csv`, and `test.csv`;
images remain in `archive` and are never copied, moved, or overwritten. Running
the script again safely regenerates the same manifests.

## Training

```bash
# Random Forest; runs handcrafted and wavelet experiments
python random_forest.py

# LightGBM; runs handcrafted and wavelet experiments
python lightgbm_model.py

# Linear supervised baseline; runs handcrafted and wavelet experiments
python logistic_regression.py

```

Each command runs a parameter grid search only on the training set with stratified cross-validation,
reports validation and final test metrics, and writes the fitted model, full search
table, metrics JSON, confusion matrices, and per-parameter performance plots under
`outputs/<model>/<features>/`.

The primary selection metric is macro F1. Accuracy and balanced accuracy are
also reported. All model files run without arguments and evaluate both feature
representations in sequence. Experiment settings such as `FEATURE_NAMES`,
`CV_FOLDS`, and `SEARCH_JOBS` are constants near the top of each model file.
`SEARCH_JOBS=4` runs four CV fits concurrently
on the M4 CPU. The test set is loaded only after cross-validation has selected
the final model.

During tuning, each model displays a dependency-free progress bar for all CV fits,
for example `Random Forest CV: 73/120` or `LightGBM CV: 156/240`.

## Files

- `prepare_dataset.py`: duplicate-safe, reproducible train/validation/test manifests
- `feature_extraction_handcrafted.py`: intensity histogram, LBP, and GLCM features
- `feature_extraction_hog.py`: HOG features
- `feature_extraction_wavelet.py`: multi-scale wavelet sub-band statistics
- `training_utils.py`: shared validation, loading, metrics, and plotting
- `random_forest.py`: grid-searched Random Forest tuning and evaluation
- `lightgbm_model.py`: grid-searched LightGBM tuning and evaluation
- `logistic_regression.py`: scaled, grid-searched linear baseline
- `CHANGELOG.md`: detailed record and rationale for all modifications

## Reproducibility notes

All random operations use seed 42. Exact duplicate content is assigned only once,
so it cannot cross train, validation, or test boundaries. Neither validation nor
test is used to choose among cross-validation candidates. Dependencies use
portable version ranges rather than machine-specific local package paths.
