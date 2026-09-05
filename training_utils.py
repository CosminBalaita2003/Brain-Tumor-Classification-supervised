"""Shared loading, evaluation, plotting, and reporting utilities."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

matplotlib_cache = Path(tempfile.gettempdir()) / "brain_tumor_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

CLASS_NAMES = ["not_tumor", "tumor"]
LABELS = list(range(len(CLASS_NAMES)))


@contextmanager
def tqdm_joblib(*, total: int, description: str):
    """Display dependency-free progress for jobs completed by sklearn/joblib."""
    completed = 0
    started = time.perf_counter()
    original_callback = joblib.parallel.BatchCompletionCallBack

    def display():
        fraction = min(completed / total, 1.0) if total else 1.0
        filled = round(30 * fraction)
        elapsed = time.perf_counter() - started
        bar = "#" * filled + "-" * (30 - filled)
        print(
            f"\r{description}: [{bar}] {completed}/{total} fits "
            f"({fraction:6.1%}, {elapsed:.0f}s)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    class ProgressCallback(original_callback):
        def __call__(self, *args, **kwargs):
            nonlocal completed
            completed += self.batch_size
            display()
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = ProgressCallback
    display()
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        if completed < total:
            completed = total
            display()
        print(file=sys.stderr)


def get_extractor(feature_name: str) -> Callable[[str], np.ndarray]:
    if feature_name == "handcrafted":
        from feature_extraction_handcrafted import extract_features
    elif feature_name == "hog":
        from feature_extraction_hog import extract_features
    elif feature_name == "wavelet":
        from feature_extraction_wavelet import extract_features
    else:
        raise ValueError(f"Unsupported feature type: {feature_name}")
    return extract_features


def build_xy(csv_path: Path, image_dir: Path, extractor: Callable[[str], np.ndarray]):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    df = pd.read_csv(csv_path)
    required = {"image", "label"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f"{csv_path} must be non-empty and contain {sorted(required)}")
    y = pd.to_numeric(df["label"], errors="raise").to_numpy(dtype=np.int64)
    unknown = sorted(set(y) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown}")
    features = []
    for image_name in df["image"].astype(str):
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"CSV references missing image: {image_path}")
        features.append(extractor(str(image_path)))
    X = np.vstack(features).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features generated for {csv_path}")
    return X, y


def metrics_block(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def plot_confusion(matrix, title: str, output_path: Path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(title=title, xlabel="Predicted label", ylabel="True label",
           xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(val_metrics: dict, test_metrics: dict, output_path: Path):
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Balanced accuracy", "Macro F1"]
    x, width = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [val_metrics[k] for k in keys], width, label="Validation")
    ax.bar(x + width / 2, [test_metrics[k] for k in keys], width, label="Test")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), title="Validation vs test metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_grid_search_parameters(cv_results: dict, param_grid: dict, output_dir: Path):
    """Plot the best mean CV macro-F1 obtained for every grid value."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(cv_results)
    score_column = "mean_test_macro_f1"
    for parameter, configured_values in param_grid.items():
        parameter_column = f"param_{parameter}"
        display_name = parameter.rsplit("__", maxsplit=1)[-1]
        labels = [str(value) for value in configured_values]
        observed = results[parameter_column].astype(str)
        scores = [
            float(results.loc[observed == label, score_column].max())
            for label in labels
        ]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, scores, marker="o")
        ax.set(
            title=f"Grid search: macro-F1 vs {display_name}",
            xlabel=display_name,
            ylabel="Best mean CV macro-F1",
            xticks=x,
            xticklabels=labels,
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"macro_f1_vs_{display_name}.png", dpi=200)
        plt.close(fig)


def save_json(payload: dict, output_path: Path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
