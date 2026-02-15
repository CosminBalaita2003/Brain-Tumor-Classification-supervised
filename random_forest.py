import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# from feature_extraction_hog import extract_features
from feature_extraction_handcrafted import extract_features


#config folders
DATA_ROOT = "dataset_brain_tumor"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "val.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test.csv")

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
VAL_DIR   = os.path.join(DATA_ROOT, "val_images")
TEST_DIR  = os.path.join(DATA_ROOT, "test_images")

RANDOM_STATE = 42

MODEL_NAME = "rf"
FEATURE_NAME = "handcrafted"
# FEATURE_NAME = "hog"

OUT_BASE = os.path.join("outputs", MODEL_NAME, FEATURE_NAME)
DIR_MODEL = os.path.join(OUT_BASE, "model")
DIR_PLOTS = os.path.join(OUT_BASE, "plots")
DIR_RESULTS = os.path.join(OUT_BASE, "results")
os.makedirs(DIR_MODEL, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)
os.makedirs(DIR_RESULTS, exist_ok=True)

LOG_PATH = os.path.join(DIR_RESULTS, "run_log.txt")
METRICS_PATH = os.path.join(DIR_RESULTS, "metrics.json")


#for logging messages to both console and file
def log(msg: str):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


#data loading + feature extraction
def build_xy(csv_path: str, img_dir: str):
    df = pd.read_csv(csv_path)
    X = []
    y = df["label"].astype(int).values

    for name in df["image"].values:
        p = os.path.join(img_dir, name)
        X.append(extract_features(p))

    X = np.vstack(X).astype(np.float32)
    return X, y, df


#plot confusion matrix with counts
def plot_confusion(cm, title, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

#plot curve of best metric per value of a parameter
def plot_param_curve(x, y, xlabel, ylabel, title, out_path, xticks=None, xtick_labels=None):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if xticks is not None and xtick_labels is not None:
        plt.xticks(xticks, xtick_labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# get best metric value for a specific parameter value from cv results
def best_metric_for_value(results_df: pd.DataFrame, param_name: str, value, metric_col: str):

    col = f"param_{param_name}"
    if value is None:
        sub = results_df[results_df[col].isna()]
    else:
        sub = results_df[results_df[col] == value]
    return float(sub[metric_col].max())



# Pick best model by VAL among top-K CV configs
def pick_best_by_val(results_df: pd.DataFrame, X_train, y_train, X_val, y_val, top_k=5, rank_col="rank_test_f1_macro"):
    if rank_col not in results_df.columns:
        raise ValueError(f"cv_results_ does not contain {rank_col}")

    candidates = results_df.sort_values(rank_col).head(top_k)

    rows = []
    best_model = None
    best_val_f1 = -1.0

    for _, r in candidates.iterrows():
        params = r["params"]

        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            **params
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        rep = classification_report(y_val, pred, digits=4, output_dict=True)

        val_f1 = float(f1_score(y_val, pred, average="macro"))
        val_acc = float(accuracy_score(y_val, pred))
        val_bacc = float(balanced_accuracy_score(y_val, pred))
        val_prec = float(rep["macro avg"]["precision"])
        val_rec  = float(rep["macro avg"]["recall"])

        rows.append({
            "params": params,
            "cv_mean_f1_macro": float(r.get("mean_test_f1_macro", np.nan)),
            "cv_mean_acc": float(r.get("mean_test_acc", np.nan)),
            "cv_mean_bal_acc": float(r.get("mean_test_bal_acc", np.nan)),
            "cv_mean_prec_macro": float(r.get("mean_test_prec_macro", np.nan)),
            "cv_mean_rec_macro": float(r.get("mean_test_rec_macro", np.nan)),
            "val_f1_macro": val_f1,
            "val_balanced_acc": val_bacc,
            "val_acc": val_acc,
            "val_prec_macro": val_prec,
            "val_rec_macro": val_rec,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model

    val_table = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    return best_model, val_table

# Compute multiple metrics and return as a dict (including confusion matrix and classification report)
def metrics_block(y_true, y_pred):
    rep = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "report": rep,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# Plot bar chart comparing VAL and TEST metrics
def plot_final_metrics_bar(val_metrics: dict, test_metrics: dict | None, out_path: str):
    labels = ["Accuracy", "Balanced Acc", "Macro Precision", "Macro Recall", "Macro F1"]
    val_vals = [
        val_metrics["accuracy"],
        val_metrics["balanced_accuracy"],
        val_metrics["macro_precision"],
        val_metrics["macro_recall"],
        val_metrics["macro_f1"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, val_vals, width, label="VAL")

    if test_metrics is not None:
        test_vals = [
            test_metrics["accuracy"],
            test_metrics["balanced_accuracy"],
            test_metrics["macro_precision"],
            test_metrics["macro_recall"],
            test_metrics["macro_f1"],
        ]
        plt.bar(x + width/2, test_vals, width, label="TEST")

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Final metrics (VAL vs TEST)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


#main function to run
def main():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"Run started: {time.ctime()}\n")

    log(f"MODEL={MODEL_NAME}  FEATURE={FEATURE_NAME}")
    log("Loading + extracting features...")

    t0 = time.time()
    X_train, y_train, _ = build_xy(TRAIN_CSV, TRAIN_DIR)
    X_val, y_val, _     = build_xy(VAL_CSV, VAL_DIR)
    feat_time = time.time() - t0

    log(f"Train shape: {X_train.shape} | Val shape: {X_val.shape}")
    log(f"Train label counts: {np.bincount(y_train)}")
    log(f"Val label counts:   {np.bincount(y_val)}")
    log(f"Feature extraction time: {feat_time:.2f}s")

    # grid search with multiple metrics
    base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )

    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 20, 30, 40],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 6, 10],
    }


    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "f1_macro": "f1_macro",
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "prec_macro": "precision_macro",
        "rec_macro": "recall_macro",
    }


    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    log("Running GridSearchCV...")
    t1 = time.time()
    gs.fit(X_train, y_train)
    gs_time = time.time() - t1

    log(f"GridSearch best params (CV, by f1_macro): {gs.best_params_}")
    log(f"GridSearch best CV f1_macro: {gs.best_score_:.6f}")
    log(f"GridSearch time: {gs_time:.2f}s")

    # save grid results
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(os.path.join(DIR_RESULTS, "gridsearch_results.csv"), index=False)
    log("Saved results: gridsearch_results.csv")

    # pick final model by VAL among top-K CV candidates
    best_rf, val_table = pick_best_by_val(
        results_df, X_train, y_train, X_val, y_val, top_k=5, rank_col="rank_test_f1_macro"
    )
    val_table.to_csv(os.path.join(DIR_RESULTS, "top5_cv_candidates_scored_on_val.csv"), index=False)
    log("Saved results: top5_cv_candidates_scored_on_val.csv")

    log("Top candidates by VAL macro-F1:")
    log(str(val_table[["val_f1_macro", "val_balanced_acc", "val_acc", "val_prec_macro", "val_rec_macro", "params"]].head(5)))


    # Validation metrics + plot
    val_pred = best_rf.predict(X_val)
    val_metrics = metrics_block(y_val, val_pred)

    cm_val = np.array(val_metrics["confusion_matrix"])
    plot_confusion(
        cm_val,
        title="Confusion Matrix (Validation) - RF with hog",
        out_path=os.path.join(DIR_PLOTS, "confusion_matrix_val.png")
    )

    # Test metrics + plot
    test_metrics = None
    if os.path.exists(TEST_CSV) and os.path.exists(TEST_DIR):
        X_test, y_test, _ = build_xy(TEST_CSV, TEST_DIR)
        test_pred = best_rf.predict(X_test)
        test_metrics = metrics_block(y_test, test_pred)

        cm_test = np.array(test_metrics["confusion_matrix"])
        plot_confusion(
            cm_test,
            title="Confusion Matrix (Test) - RF with hog",
            out_path=os.path.join(DIR_PLOTS, "confusion_matrix_test.png")
        )

    plot_final_metrics_bar(
        val_metrics,
        test_metrics,
        out_path=os.path.join(DIR_PLOTS, "final_metrics_val_vs_test.png")
    )

    # Plot curves (BEST CV per value) for multiple metrics
    metric_cols = {
        "f1_macro": ("mean_test_f1_macro", "CV macro-F1"),
        "acc": ("mean_test_acc", "CV Accuracy"),
        "bal_acc": ("mean_test_bal_acc", "CV Balanced Acc"),
        "prec_macro": ("mean_test_prec_macro", "CV Macro Precision"),
        "rec_macro": ("mean_test_rec_macro", "CV Macro Recall"),
    }

    def plot_all_metrics_for_param(param_name, values, labels=None):
        # x axis
        if labels is None:
            x = values
            xticks = None
            xtick_labels = None
        else:
            x = list(range(len(values)))
            xticks = x
            xtick_labels = labels

        for key, (col, ylabel) in metric_cols.items():
            y = [best_metric_for_value(results_df, param_name, v, col) for v in values]
            plot_param_curve(
                x, y,
                xlabel=param_name,
                ylabel=ylabel,
                title=f"RF: BEST {ylabel} vs {param_name}",
                out_path=os.path.join(DIR_PLOTS, f"cv_best_{key}_vs_{param_name}.png"),
                xticks=xticks,
                xtick_labels=xtick_labels
            )

    # max_depth (needs special tick labels for None)
    md_vals = [None, 20, 30, 40]
    md_labels = ["None", "20", "30", "40"]
    plot_all_metrics_for_param("max_depth", md_vals, labels=md_labels)

    # n_estimators
    plot_all_metrics_for_param("n_estimators", [200, 400, 800])

    # min_samples_leaf
    plot_all_metrics_for_param("min_samples_leaf", [1, 2, 4])

    # min_samples_split
    plot_all_metrics_for_param("min_samples_split", [2, 6, 10])

    # Save model
    model_path = os.path.join(DIR_MODEL, "rf_final.pkl")
    joblib.dump(best_rf, model_path)
    log(f"Saved model: {model_path}")

    # Save metrics summary (JSON)
    summary = {
        "model": MODEL_NAME,
        "feature": FEATURE_NAME,
        "random_state": RANDOM_STATE,
        "feature_extraction_time_sec": float(feat_time),
        "gridsearch_time_sec": float(gs_time),
        "grid_best_params_cv": gs.best_params_,
        "grid_best_cv_f1_macro": float(gs.best_score_),
        "val": val_metrics,
        "test": test_metrics
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metrics: {METRICS_PATH}")

    log("Done")


if __name__ == "__main__":
    main()
