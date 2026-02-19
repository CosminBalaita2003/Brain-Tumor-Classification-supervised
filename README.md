# Brain Tumor Classification using Handcrafted and HOG Features

## Overview

This project implements a supervised machine learning pipeline for multi-class brain tumor image classification. The objective is to classify MRI images into four categories using classical machine learning models and two different feature representations.

The project follows a complete ML workflow:
- Dataset preparation
- Feature extraction
- Model training
- Hyperparameter tuning
- Evaluation on validation and test sets
- Performance visualization

---

## Dataset

The dataset contains MRI brain images grouped into four classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

The original dataset is split into:
- Training set
- Validation set (created from training using stratified split)
- Test set (used only for final evaluation)

Stratified splitting is used to preserve class distribution across subsets.

---

## Feature Extraction

Two different feature representations are implemented.

### 1. Handcrafted Features

This representation combines:

- Global intensity statistics (mean, standard deviation, skewness, kurtosis)
- Intensity histogram
- Local Binary Patterns (LBP)
- Gray Level Co-occurrence Matrix (GLCM) texture properties

These features capture intensity distribution and texture patterns.

### 2. HOG Features (Histogram of Oriented Gradients)

HOG extracts structural and edge-based information by:
- Computing image gradients
- Building orientation histograms in local regions
- Normalizing feature blocks

This representation focuses on shape and contour information.

---

## Models

Two supervised learning models are implemented.

### 1. k-Nearest Neighbors (kNN)

- Distance-based classifier
- Uses Minkowski distance (Euclidean or Manhattan)
- Supports uniform and distance-based weighting
- Requires feature scaling

### 2. Random Forest

- Ensemble of decision trees
- Uses Gini impurity for splitting
- Includes built-in regularization through:
  - Maximum depth
  - Minimum samples per split
  - Minimum samples per leaf
- Reduces overfitting via bagging and random feature selection

---

## Hyperparameter Tuning

Hyperparameters are optimized using GridSearchCV with Stratified K-Fold cross-validation.

Examples of tuned parameters:

kNN:
- Number of neighbors
- Distance metric
- Weighting scheme
- Minkowski power parameter

Random Forest:
- Number of trees
- Maximum depth
- Minimum samples split
- Minimum samples leaf
- Maximum features

Model selection is based on validation performance.

---

## Evaluation Metrics

The following metrics are computed:

- Accuracy
- Macro F1-score
- Balanced accuracy
- Confusion matrix

Macro F1 is used to give equal importance to each class.

Balanced accuracy compensates for possible class imbalance.

---

## Visualizations

The project generates multiple plots:

- Validation macro F1 vs number of neighbors (kNN)
- Cross-validation score vs max depth (Random Forest)
- Cross-validation score vs number of estimators (Random Forest)
- Confusion matrices for validation and test sets

These plots help analyze:
- Overfitting vs underfitting behavior
- Model stability
- Hyperparameter sensitivity

---

## Project Structure
- prepare_dataset.py
- feature_extraction_handcrafted.py
- feature_extraction_hog.py
- knn.py
- random_forest.py
- outputs/


- `prepare_dataset.py` handles data splitting.
- Feature extraction files compute image representations.
- Model files train, tune, evaluate and save results.
- Outputs contain plots, confusion matrices and result summaries.

---

## Experimental Setup

All experiments:
- Use fixed random seeds for reproducibility
- Perform hyperparameter tuning only on training data
- Use validation set for model selection
- Evaluate final performance on unseen test data

---

## Conclusion

The project demonstrates a complete classical machine learning pipeline for medical image classification. It compares:

- Two feature representations
- Two supervised models

The comparison is performed using consistent evaluation methodology and cross-validation, ensuring fair and reproducible results.
