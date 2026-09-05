"""Tune and evaluate Logistic Regression as a linear supervised baseline."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training_utils import (build_xy, get_extractor, metrics_block, plot_confusion,
                            plot_grid_search_parameters, plot_metric_comparison,
                            save_json, tqdm_joblib)

RANDOM_STATE = 42
FEATURE_NAMES = ("handcrafted", "wavelet")
CV_FOLDS = 5
SEARCH_JOBS = 4
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset_brain_tumor"


def load_split(name, extractor):
    return build_xy(DATA_ROOT / f"{name}.csv", PROJECT_ROOT, extractor)


def run_experiment(feature_name: str):
    output = PROJECT_ROOT / "outputs" / "logistic_regression" / feature_name
    model_dir, plot_dir, result_dir = output / "model", output / "plots", output / "results"
    for directory in (model_dir, plot_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    extractor = get_extractor(feature_name)
    started = time.perf_counter()
    X_train, y_train = load_split("train", extractor)
    X_val, y_val = load_split("val", extractor)
    min_class_count = int(np.bincount(y_train).min())
    if CV_FOLDS > min_class_count:
        raise ValueError(f"CV_FOLDS cannot exceed smallest training class ({min_class_count})")

    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear", class_weight="balanced",
            random_state=RANDOM_STATE, max_iter=5000,
        )),
    ])
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidate_count = len(ParameterGrid(param_grid))
    search = GridSearchCV(
        estimator, param_grid,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                 "accuracy": "accuracy"},
        refit="macro_f1", cv=cv, n_jobs=SEARCH_JOBS,
        verbose=1, return_train_score=True,
    )
    with tqdm_joblib(total=candidate_count * CV_FOLDS,
                     description=f"Logistic Regression CV ({feature_name})"):
        search.fit(X_train, y_train)
    pd.DataFrame(search.cv_results_).to_csv(result_dir / "search_results.csv", index=False)
    plot_grid_search_parameters(search.cv_results_, param_grid, plot_dir / "hyperparameters")

    best_model = search.best_estimator_
    val_metrics = metrics_block(y_val, best_model.predict(X_val))
    plot_confusion(val_metrics["confusion_matrix"],
                   f"Logistic Regression validation ({feature_name})",
                   plot_dir / "confusion_matrix_val.png")

    X_test, y_test = load_split("test", extractor)
    test_metrics = metrics_block(y_test, best_model.predict(X_test))
    plot_confusion(test_metrics["confusion_matrix"],
                   f"Logistic Regression test ({feature_name})",
                   plot_dir / "confusion_matrix_test.png")
    plot_metric_comparison(val_metrics, test_metrics, plot_dir / "metrics_val_vs_test.png")
    joblib.dump(best_model, model_dir / "logistic_regression.joblib")
    save_json({
        "model": "logistic_regression",
        "role": "linear_supervised_baseline",
        "features": feature_name,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "search_type": "grid_search",
        "grid_candidates": candidate_count,
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }, result_dir / "metrics.json")

    print(f"[{feature_name}] Best CV macro-F1: {search.best_score_:.4f}")
    print(f"[{feature_name}] Validation macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"[{feature_name}] Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"[{feature_name}] Artifacts saved under {output}")


def main():
    for feature_name in FEATURE_NAMES:
        run_experiment(feature_name)


if __name__ == "__main__":
    main()
