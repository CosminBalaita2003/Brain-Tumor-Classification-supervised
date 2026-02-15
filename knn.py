import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# from feature_extraction_hog import extract_features
from feature_extraction_handcrafted import extract_features


# config
DATA_ROOT = "dataset_brain_tumor"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "val.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test.csv")

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
VAL_DIR   = os.path.join(DATA_ROOT, "val_images")
TEST_DIR  = os.path.join(DATA_ROOT, "test_images")

RANDOM_STATE = 42

MODEL_NAME = "knn"

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


#plots
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


def plot_metric_bars(metrics_val: dict, metrics_test: dict, out_path: str):

    names = ["accuracy", "balanced_accuracy", "macro_f1"]
    val_vals = [metrics_val.get(n, np.nan) for n in names]
    test_vals = [metrics_test.get(n, np.nan) for n in names] if metrics_test else [np.nan]*len(names)

    x = np.arange(len(names))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, val_vals, width, label="VAL")
    plt.bar(x + width/2, test_vals, width, label="TEST")
    plt.xticks(x, names, rotation=0)
    plt.ylim(0, 1)
    plt.title("Summary metrics (VAL vs TEST)")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_class_prf(report_dict: dict, split_name: str, out_path: str):
    class_keys = [k for k in report_dict.keys() if k.isdigit()]
    class_keys = sorted(class_keys, key=int)

    prec = [report_dict[k]["precision"] for k in class_keys]
    rec  = [report_dict[k]["recall"] for k in class_keys]
    f1   = [report_dict[k]["f1-score"] for k in class_keys]

    x = np.arange(len(class_keys))
    width = 0.28

    plt.figure()
    plt.bar(x - width, prec, width, label="precision")
    plt.bar(x,         rec,  width, label="recall")
    plt.bar(x + width, f1,   width, label="f1")
    plt.xticks(x, class_keys)
    plt.ylim(0, 1)
    plt.title(f"Per-class PRF ({split_name})")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def best_score_for_value(results_df: pd.DataFrame, param_name: str, value):

    col = f"param_{param_name}"
    sub = results_df[results_df[col] == value]
    return float(sub["mean_test_score"].max())



def pick_best_by_val(results_df: pd.DataFrame, X_train, y_train, X_val, y_val, top_k=5):
    if "rank_test_score" not in results_df.columns:
        raise ValueError("cv_results_ does not contain rank_test_score")

    candidates = results_df.sort_values("rank_test_score").head(top_k)

    rows = []
    best_model = None
    best_val_f1 = -1.0

    for _, r in candidates.iterrows():
        params = r["params"]

        # reconstruim pipeline-ul si aplicam params
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_val)
        val_f1 = f1_score(y_val, pred, average="macro")
        val_acc = accuracy_score(y_val, pred)
        val_bacc = balanced_accuracy_score(y_val, pred)

        rows.append({
            "params": params,
            "cv_mean_f1_macro": float(r["mean_test_score"]),
            "val_f1_macro": float(val_f1),
            "val_balanced_acc": float(val_bacc),
            "val_acc": float(val_acc),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = pipe

    val_table = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    return best_model, val_table


def metrics_block(y_true, y_pred):
    rep = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": rep,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }



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


    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": [ 3, 5, 7, 9, 11, 15, 21, 31],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski", "euclidean", "manhattan"],
        "knn__p": [1, 2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    log("Running GridSearchCV...")
    t1 = time.time()
    gs.fit(X_train, y_train)
    gs_time = time.time() - t1

    log(f"GridSearch best params (CV): {gs.best_params_}")
    log(f"GridSearch best CV macro-F1: {gs.best_score_:.6f}")
    log(f"GridSearch time: {gs_time:.2f}s")

    # save grid results
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(os.path.join(DIR_RESULTS, "gridsearch_results.csv"), index=False)
    log("Saved results: gridsearch_results.csv")

    best_knn, val_table = pick_best_by_val(results_df, X_train, y_train, X_val, y_val, top_k=5)
    val_table.to_csv(os.path.join(DIR_RESULTS, "top5_cv_candidates_scored_on_val.csv"), index=False)
    log("Saved results: top5_cv_candidates_scored_on_val.csv")

    log("Top candidates by VAL macro-F1:")
    log(str(val_table[["val_f1_macro", "val_balanced_acc", "val_acc", "cv_mean_f1_macro", "params"]].head(5)))


    val_pred = best_knn.predict(X_val)
    val_metrics = metrics_block(y_val, val_pred)

    cm_val = np.array(val_metrics["confusion_matrix"])
    plot_confusion(
        cm_val,
        title="Confusion Matrix (Validation) - KNN FINAL",
        out_path=os.path.join(DIR_PLOTS, "confusion_matrix_val.png")
    )

    plot_per_class_prf(
        val_metrics["report"],
        split_name="VAL",
        out_path=os.path.join(DIR_PLOTS, "per_class_prf_val.png")
    )


    k_vals = [1, 3, 5, 7, 9, 11, 15, 21]
    k_best = [best_score_for_value(results_df, "knn__n_neighbors", k) for k in k_vals]
    plot_param_curve(
        k_vals, k_best,
        xlabel="n_neighbors (k)",
        ylabel="CV macro-F1 (BEST)",
        title="KNN: BEST CV macro-F1 vs n_neighbors",
        out_path=os.path.join(DIR_PLOTS, "cv_best_vs_k.png")
    )


    w_vals = ["uniform", "distance"]
    w_best = [best_score_for_value(results_df, "knn__weights", w) for w in w_vals]
    x = list(range(len(w_vals)))
    plot_param_curve(
        x, w_best,
        xlabel="weights",
        ylabel="CV macro-F1 (BEST)",
        title="KNN: BEST CV macro-F1 vs weights",
        out_path=os.path.join(DIR_PLOTS, "cv_best_vs_weights.png"),
        xticks=x,
        xtick_labels=w_vals
    )

    m_vals = ["minkowski", "euclidean", "manhattan"]
    m_best = [best_score_for_value(results_df, "knn__metric", m) for m in m_vals]
    x = list(range(len(m_vals)))
    plot_param_curve(
        x, m_best,
        xlabel="metric",
        ylabel="CV macro-F1 (BEST)",
        title="KNN: BEST CV macro-F1 vs metric",
        out_path=os.path.join(DIR_PLOTS, "cv_best_vs_metric.png"),
        xticks=x,
        xtick_labels=m_vals
    )

    p_vals = [1, 2]
    p_best = [best_score_for_value(results_df, "knn__p", p) for p in p_vals]
    plot_param_curve(
        p_vals, p_best,
        xlabel="p (Minkowski)",
        ylabel="CV macro-F1 (BEST)",
        title="KNN: BEST CV macro-F1 vs p",
        out_path=os.path.join(DIR_PLOTS, "cv_best_vs_p.png")
    )


    test_metrics = None
    if os.path.exists(TEST_CSV) and os.path.exists(TEST_DIR):
        X_test, y_test, _ = build_xy(TEST_CSV, TEST_DIR)
        test_pred = best_knn.predict(X_test)
        test_metrics = metrics_block(y_test, test_pred)

        cm_test = np.array(test_metrics["confusion_matrix"])
        plot_confusion(
            cm_test,
            title="Confusion Matrix (Test) - KNN FINAL",
            out_path=os.path.join(DIR_PLOTS, "confusion_matrix_test.png")
        )

        plot_per_class_prf(
            test_metrics["report"],
            split_name="TEST",
            out_path=os.path.join(DIR_PLOTS, "per_class_prf_test.png")
        )

        plot_metric_bars(
            val_metrics,
            test_metrics,
            out_path=os.path.join(DIR_PLOTS, "summary_metrics_val_vs_test.png")
        )


    model_path = os.path.join(DIR_MODEL, "knn_final.pkl")
    joblib.dump(best_knn, model_path)
    log(f"Saved model: {model_path}")


    summary = {
        "model": MODEL_NAME,
        "feature": FEATURE_NAME,
        "random_state": RANDOM_STATE,
        "feature_extraction_time_sec": float(feat_time),
        "gridsearch_time_sec": float(gs_time),
        "grid_best_params_cv": gs.best_params_,
        "grid_best_cv_macro_f1": float(gs.best_score_),
        "val": val_metrics,
        "test": test_metrics
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metrics: {METRICS_PATH}")

    log("Done")


if __name__ == "__main__":
    main()
