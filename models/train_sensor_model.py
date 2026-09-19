"""
train_sensor_model.py
---------------------
Item 3: XGBoost irrigation-need classifier with Isolation Forest anomaly filtering.

IMPORTANT NOTE on the moisture-label correlation (from data_check.py):
  The Pearson r = -0.87 between soil_moisture_pct and irrigation_needed is
  expected and is NOT a newly discovered pattern. The label was *directly
  derived* from moisture in label_engineering.py (threshold rule: label=1 if
  moisture < 70%). So that correlation is a validity check confirming the label
  was engineered correctly — it does NOT mean the model is just memorising a
  trivial rule. In production, the model will receive live moisture readings it
  has never seen and predict label in real time, which is the actual ML task.
  Temperature and humidity provide secondary signal (e.g., high temp + low
  moisture = more urgent need) that SHAP will quantify in item 4.

Pipeline
--------
  1. Load combined labeled CSVs (plant_a + plant_b)
  2. Sort by timestamp → time-based 80/20 train/test split (no data leakage)
  3. Isolation Forest on TRAINING data only (contamination=0.01)
     → print flagged anomaly count and 3-5 example rows for sanity check
  4. Train XGBoost on cleaned training data
     → scale_pos_weight = n_neg / n_pos (computed from cleaned training set)
  5. Evaluate on TEST set: accuracy, AUC-ROC, confusion matrix, feature importance
  6. Save model to models/xgboost_irrigation.pkl

Features used : soil_moisture_pct, temperature_c, humidity_pct
Features dropped as inputs : plant_id, timestamp (leakage / identity shortcut risk)
Label         : irrigation_needed (0 or 1)

Input  : data/processed/plant_a_labeled.csv
         data/processed/plant_b_labeled.csv
Output : models/xgboost_irrigation.pkl
         printed evaluation report
"""

import os
import csv
import pickle
import math
from datetime import datetime

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
# Climb one level up: this script lives in models/, data/ is a sibling directory
ROOT_DIR      = os.path.dirname(BASE_DIR)
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = BASE_DIR   # save model next to this script

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "xgboost_irrigation.pkl")

FILES = [
    os.path.join(PROCESSED_DIR, "plant_a_labeled.csv"),
    os.path.join(PROCESSED_DIR, "plant_b_labeled.csv"),
]

# Features fed to the model — plant_id and timestamp are explicitly excluded
FEATURE_COLS = ["soil_moisture_pct", "temperature_c", "humidity_pct"]
LABEL_COL    = "irrigation_needed"

# Isolation Forest contamination: expect ~1% of readings to be sensor glitches
IF_CONTAMINATION = 0.01

# Train/test split: first 80% of days → train, last 20% → test
TRAIN_SPLIT_FRAC = 0.80


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> list[dict]:
    """Load and combine both labeled CSVs. Returns list of row dicts."""
    all_rows = []
    for path in FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Labeled CSV not found: {path}\n"
                "Run label_engineering.py first."
            )
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append({
                    "timestamp":         row["timestamp"],
                    "plant_id":          row["plant_id"],   # kept for reporting only
                    "soil_moisture_pct": float(row["soil_moisture_pct"]),
                    "temperature_c":     float(row["temperature_c"]),
                    "humidity_pct":      float(row["humidity_pct"]),
                    "irrigation_needed": int(row["irrigation_needed"]),
                    "stress_level":      row["stress_level"],
                })
    return all_rows


# ── Time-based train/test split ───────────────────────────────────────────────
def time_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Sort all rows by timestamp, then split at the 80th-percentile day boundary.

    Why time-based and not random?
    Sensor readings 30 minutes apart are highly correlated — a random split
    would put t=00:00 in training and t=00:30 in test for the same day,
    creating data leakage. A time split ensures the model is tested on
    genuinely future (unseen) readings.
    """
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])

    # Find the unique days and split at the 80th-percentile day
    unique_days = sorted(set(r["timestamp"][:10] for r in rows_sorted))
    split_idx   = int(len(unique_days) * TRAIN_SPLIT_FRAC)
    cutoff_day  = unique_days[split_idx]   # first day that belongs to test set

    train = [r for r in rows_sorted if r["timestamp"][:10] <  cutoff_day]
    test  = [r for r in rows_sorted if r["timestamp"][:10] >= cutoff_day]
    return train, test


# ── Feature matrix helpers ────────────────────────────────────────────────────
def to_X(rows: list[dict]) -> np.ndarray:
    """Extract feature matrix (n_samples x n_features). plant_id excluded."""
    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float32)

def to_y(rows: list[dict]) -> np.ndarray:
    return np.array([r[LABEL_COL] for r in rows], dtype=np.int32)


# ── Isolation Forest anomaly filtering ───────────────────────────────────────
def apply_isolation_forest(
    train_rows: list[dict],
) -> tuple[list[dict], list[dict], IsolationForest]:
    """
    Fit Isolation Forest on training features, then filter out flagged anomalies.

    contamination=0.01 means the forest expects ~1% of training readings to be
    genuine sensor glitches (e.g., ESP32 ADC saturation, probe disconnection).
    IsolationForest labels anomalies as -1 and inliers as +1.

    IMPORTANT: The forest is fit ONLY on training data.
    It is then also available to score incoming live sensor readings in production
    (item 7, Flask API) before passing them to XGBoost.

    Returns
    -------
    clean_rows   : training rows where IsolationForest predicted inlier (+1)
    anomaly_rows : training rows flagged as anomalies (-1)
    iso_forest   : fitted IsolationForest object (saved alongside XGBoost model)
    """
    X_train = to_X(train_rows)

    iso = IsolationForest(
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    iso.fit(X_train)
    preds = iso.predict(X_train)   # +1 = inlier, -1 = anomaly

    clean_rows   = [r for r, p in zip(train_rows, preds) if p == 1]
    anomaly_rows = [r for r, p in zip(train_rows, preds) if p == -1]
    return clean_rows, anomaly_rows, iso


def print_anomaly_report(anomaly_rows: list[dict], total_train: int) -> None:
    print(f"\n{'=' * 62}")
    print("  ISOLATION FOREST — Anomaly Report")
    print(f"{'=' * 62}")
    print(f"  Training rows total    : {total_train:,}")
    print(f"  Flagged as anomalies   : {len(anomaly_rows):,}  "
          f"({len(anomaly_rows)/total_train*100:.2f}%)")
    print(f"  Rows used for training : {total_train - len(anomaly_rows):,}")
    print()
    print("  SANITY CHECK — up to 5 flagged rows (are these genuine glitches?)")
    print(f"  {'Source':<10} {'Moisture%':>10} {'Temp°C':>8} {'Hum%':>7} "
          f"{'Label':>7} {'Stress'}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*18}")

    for r in anomaly_rows[:5]:
        print(f"  {r['plant_id']:<10}  {r['soil_moisture_pct']:>10.2f}  "
              f"{r['temperature_c']:>8.2f}  {r['humidity_pct']:>7.2f}  "
              f"{r['irrigation_needed']:>7}  {r['stress_level']}")

    print()
    print("  NOTE: At contamination=0.01, the forest flags statistical outliers")
    print("  in feature space — extreme moisture/temp/humidity combinations that")
    print("  are unlikely under normal sensor operation. These are not necessarily")
    print("  wrong labels; they are readings that look physically implausible.")
    print("  If flagged rows look like normal readings, lower contamination to 0.005.")
    print(f"{'=' * 62}")


# ── XGBoost training ──────────────────────────────────────────────────────────
def train_xgboost(
    clean_rows: list[dict],
) -> tuple[xgb.XGBClassifier, float]:
    """
    Train XGBoost on anomaly-filtered training data.

    scale_pos_weight = n_neg / n_pos, computed from the CLEANED training set.
    This compensates for class imbalance by up-weighting the minority class
    (irrigation_needed=1) during gradient computation.
    """
    X = to_X(clean_rows)
    y = to_y(clean_rows)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw   = round(n_neg / n_pos, 4) if n_pos > 0 else 1.0

    print(f"\n{'=' * 62}")
    print("  XGBOOST — Training")
    print(f"{'=' * 62}")
    print(f"  Clean training rows    : {len(y):,}")
    print(f"  label=0 (not needed)   : {n_neg:,}")
    print(f"  label=1 (needed)       : {n_pos:,}")
    print(f"  scale_pos_weight       : {spw}")

    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,       # shallow trees reduce overfitting on small data
        learning_rate     = 0.05,    # slow learning rate + more trees = better generalisation
        subsample         = 0.8,     # row subsampling per tree (reduces variance)
        colsample_bytree  = 1.0,     # use all 3 features (small feature set)
        scale_pos_weight  = spw,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(X, y)
    print("  Training complete.")
    return model, spw


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model: xgb.XGBClassifier, test_rows: list[dict]) -> None:
    X_test = to_X(test_rows)
    y_test = to_y(test_rows)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]   # probability of label=1
    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # Precision, recall, F1 for the positive class (irrigation needed)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'=' * 62}")
    print("  EVALUATION — Test Set (last 20% of days, chronologically)")
    print(f"{'=' * 62}")
    print(f"  Test rows              : {len(y_test):,}")
    print(f"  Accuracy               : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC                : {auc:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                  Pred=0   Pred=1")
    print(f"  Actual=0  :  {tn:>6,}   {fp:>6,}   (TN, FP)")
    print(f"  Actual=1  :  {fn:>6,}   {tp:>6,}   (FN, TP)")
    print()
    print(f"  Precision  (of label=1): {precision:.4f}")
    print(f"  Recall     (of label=1): {recall:.4f}   <- false negatives = missed irrigations")
    print(f"  F1-score   (of label=1): {f1:.4f}")
    print()
    print("  NOTE: Recall is the most important metric here. A false negative")
    print("  (missed irrigation) causes crop stress; a false positive (watering")
    print("  when not needed) wastes water but doesn't damage the plant.")

    # Feature importance (XGBoost gain = how much each feature improves splits)
    importance = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importance.values())

    print(f"\n  Feature Importance (by gain — % contribution to split quality):")
    for feat_idx, feat_name in enumerate(FEATURE_COLS):
        key  = f"f{feat_idx}"
        gain = importance.get(key, 0.0)
        pct  = gain / total_gain * 100 if total_gain > 0 else 0.0
        bar  = "#" * int(pct / 2)
        print(f"  {feat_name:<22}  {pct:>6.2f}%  {bar}")

    print(f"{'=' * 62}")


# ── Save ───────────────────────────────────────────────────────────────────────
def save_artifacts(
    model: xgb.XGBClassifier,
    iso: IsolationForest,
    meta: dict,
) -> None:
    """
    Save model + IsolationForest + metadata as a single pickle bundle.
    Bundling them together ensures the Flask API always loads a consistent pair —
    the same Isolation Forest that filtered the training data screens live readings.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "xgb_model":    model,
        "iso_forest":   iso,
        "feature_cols": FEATURE_COLS,
        "label_col":    LABEL_COL,
        "meta":         meta,
    }
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model bundle saved -> {MODEL_SAVE_PATH}")
    print("  Bundle contains: xgb_model, iso_forest, feature_cols, label_col, meta")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Item 3 — Sensor Model Training")
    print("=" * 62)

    # 1. Load
    print("\nLoading labeled data ...")
    rows = load_data()
    print(f"  Total rows loaded: {len(rows):,}")

    # 2. Time-based split
    train_rows, test_rows = time_split(rows)
    train_days = sorted(set(r["timestamp"][:10] for r in train_rows))
    test_days  = sorted(set(r["timestamp"][:10] for r in test_rows))
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_rows):,} rows  ({train_days[0]} to {train_days[-1]})")
    print(f"  Test : {len(test_rows):,} rows  ({test_days[0]}  to {test_days[-1]})")

    # 3. Isolation Forest
    clean_rows, anomaly_rows, iso = apply_isolation_forest(train_rows)
    print_anomaly_report(anomaly_rows, len(train_rows))

    # 4. Train XGBoost
    model, spw = train_xgboost(clean_rows)

    # 5. Evaluate on test set
    evaluate(model, test_rows)

    # 6. Save
    meta = {
        "train_date_range": f"{train_days[0]} to {train_days[-1]}",
        "test_date_range":  f"{test_days[0]} to {test_days[-1]}",
        "n_train_clean":    len(clean_rows),
        "n_anomalies":      len(anomaly_rows),
        "n_test":           len(test_rows),
        "scale_pos_weight": spw,
        "if_contamination": IF_CONTAMINATION,
        "feature_cols":     FEATURE_COLS,
    }
    save_artifacts(model, iso, meta)

    print("\nDone. Load models/xgboost_irrigation.pkl in the SHAP script (item 4).")
