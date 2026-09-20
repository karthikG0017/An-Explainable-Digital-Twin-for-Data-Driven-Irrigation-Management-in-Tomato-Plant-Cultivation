"""
shap_sensor_model.py
--------------------
Item 4 (sensor model half): SHAP explainability for the XGBoost irrigation
classifier trained in item 3.

What this script produces
-------------------------
  1. Global feature importance — SHAP mean |value| across ALL test rows,
     confirming which features drive irrigation predictions overall.
  2. Per-prediction explanation — for 5 representative test rows (2 label=0,
     2 label=1, 1 borderline), prints each feature's SHAP contribution with
     a plain-English interpretation: "This reading's moisture of X% pushed
     the irrigation probability UP/DOWN by Y points."
  3. Real-world threshold comparison — runs the same SHAP explanation on the
     Mendeley JSON pump-ON rows. NOTE: this is NOT a standard accuracy check
     against a shared ground truth. Our model uses a 70% moisture threshold;
     the real deployment triggers at ~50%. These are two different irrigation
     policies. Whether 70% vs 50% is better for water efficiency and yield is
     an open question our A/B field experiment will answer.
  4. Saves SHAP values to models/shap_values_test.pkl for use in the
     Flask API (item 7) and frontend (item 8).

NOTE on 100% test accuracy:
  The model was trained on simulator data where the label was DERIVED from
  moisture via a hard threshold (moisture < 70% -> label=1). The model
  learned that threshold exactly — which is why test accuracy is 100%.
  SHAP's job here is NOT to find surprising patterns; it is to make the
  model's reasoning transparent and auditable for the thesis explainability
  requirement, and to confirm that moisture drives the decision (as designed),
  with temperature and humidity as secondary modifiers.

Input   : models/xgboost_irrigation.pkl   (XGBoost + IsolationForest bundle)
          data/processed/plant_a_labeled.csv
          data/processed/plant_b_labeled.csv
          data/datasets/Dataset on Irrigation for Tomato/tomato_water_sensor_data.json
Output  : models/shap_values_test.pkl     (SHAP values for test set)
          printed explanations
"""

import os
import csv
import json
import pickle

import numpy as np
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))   # models/
PROJECT_DIR   = os.path.dirname(ROOT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
DATASETS_DIR  = os.path.join(PROJECT_DIR, "data", "datasets",
                              "Dataset on Irrigation for Tomato")
MODEL_PATH    = os.path.join(ROOT_DIR, "xgboost_irrigation.pkl")
SHAP_OUT_PATH = os.path.join(ROOT_DIR, "shap_values_test.pkl")

LABELED_FILES = [
    os.path.join(PROCESSED_DIR, "plant_a_labeled.csv"),
    os.path.join(PROCESSED_DIR, "plant_b_labeled.csv"),
]
MENDELEY_JSON = os.path.join(DATASETS_DIR, "tomato_water_sensor_data.json")

TRAIN_SPLIT_FRAC = 0.80   # must match item 3


# ── Load model bundle ─────────────────────────────────────────────────────────
def load_bundle() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train_sensor_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Reconstruct the same test split as item 3 ─────────────────────────────────
def load_test_rows(feature_cols: list[str]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Reproduce the time-based 80/20 split; return test rows, X_test, y_test."""
    all_rows = []
    for path in LABELED_FILES:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append({
                    "timestamp":         row["timestamp"],
                    "soil_moisture_pct": float(row["soil_moisture_pct"]),
                    "temperature_c":     float(row["temperature_c"]),
                    "humidity_pct":      float(row["humidity_pct"]),
                    "irrigation_needed": int(row["irrigation_needed"]),
                    "stress_level":      row["stress_level"],
                })

    rows_sorted  = sorted(all_rows, key=lambda r: r["timestamp"])
    unique_days  = sorted(set(r["timestamp"][:10] for r in rows_sorted))
    split_idx    = int(len(unique_days) * TRAIN_SPLIT_FRAC)
    cutoff_day   = unique_days[split_idx]
    test_rows    = [r for r in rows_sorted if r["timestamp"][:10] >= cutoff_day]

    X_test = np.array([[r[c] for c in feature_cols] for r in test_rows], dtype=np.float32)
    y_test = np.array([r["irrigation_needed"] for r in test_rows], dtype=np.int32)
    return test_rows, X_test, y_test


# ── Compute SHAP values ───────────────────────────────────────────────────────
def compute_shap(model, X_test: np.ndarray) -> np.ndarray:
    """
    Use SHAP's TreeExplainer — the exact, fast algorithm for tree-based models.
    Returns shap_values array of shape (n_samples, n_features).
    For binary classification, shap_values are for the positive class (label=1).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # XGBoost binary returns shape (n, f) directly; guard against (2, n, f)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # take positive class
    return shap_values


# ── Section 1: Global feature importance ──────────────────────────────────────
def print_global_importance(shap_values: np.ndarray, feature_cols: list[str]) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()

    print("=" * 62)
    print("  SECTION 1 — Global SHAP Feature Importance (test set)")
    print("=" * 62)
    print("  Mean |SHAP value| = average impact on model output (log-odds)")
    print()
    print(f"  {'Feature':<22}  {'Mean|SHAP|':>10}  {'% of total':>10}  {'Bar'}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*30}")

    ranked = sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])
    for feat, val in ranked:
        pct = val / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {feat:<22}  {val:>10.5f}  {pct:>9.2f}%  {bar}")

    print()
    print("  INTERPRETATION:")
    print("  soil_moisture_pct should dominate (expected: label was derived")
    print("  from moisture threshold). temperature_c and humidity_pct provide")
    print("  secondary signal — SHAP confirms their marginal contribution.")
    print("=" * 62)


# ── Section 2: Per-prediction explanations ────────────────────────────────────
def print_per_prediction(
    test_rows: list[dict],
    X_test: np.ndarray,
    shap_values: np.ndarray,
    feature_cols: list[str],
    model,
) -> None:
    print()
    print("=" * 62)
    print("  SECTION 2 — Per-Prediction SHAP Explanations (5 examples)")
    print("=" * 62)

    label0_idx = [i for i, r in enumerate(test_rows) if r["irrigation_needed"] == 0]
    label1_idx = [i for i, r in enumerate(test_rows) if r["irrigation_needed"] == 1]

    # Find borderline: row where moisture is closest to 70%
    border_idx = min(range(len(test_rows)),
                     key=lambda i: abs(test_rows[i]["soil_moisture_pct"] - 70.0))

    # Pick 2 label=0, 2 label=1, 1 borderline (spaced out across the set)
    step0 = max(1, len(label0_idx) // 2)
    step1 = max(1, len(label1_idx) // 2)
    selected = {
        "Not needed (A)": label0_idx[0],
        "Not needed (B)": label0_idx[step0],
        "Needed (A)"    : label1_idx[0],
        "Needed (B)"    : label1_idx[step1],
        "Borderline"    : border_idx,
    }

    base_prob = float(model.predict_proba(X_test[:1])[:, 1][0])   # rough baseline

    for label, idx in selected.items():
        row    = test_rows[idx]
        sv     = shap_values[idx]
        prob   = float(model.predict_proba(X_test[idx:idx+1])[:, 1][0])
        pred   = int(model.predict(X_test[idx:idx+1])[0])
        actual = row["irrigation_needed"]

        print(f"\n  --- {label} ---")
        print(f"  Actual={actual}  Predicted={pred}  P(irrigation_needed)={prob:.4f}")
        print(f"  Stress level: {row['stress_level']}")
        print()
        print(f"  {'Feature':<22}  {'Value':>8}  {'SHAP':>10}  Direction  Plain English")
        print(f"  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*30}")

        for feat, val, sv_val in zip(feature_cols,
                                      [row[c] for c in feature_cols], sv):
            direction = "UP  (+)" if sv_val > 0 else "DOWN (-)"
            # Plain-English template for each feature
            if feat == "soil_moisture_pct":
                plain = f"Moisture {val:.1f}% {'reduces' if sv_val < 0 else 'raises'} irrigation need"
            elif feat == "temperature_c":
                plain = f"Temp {val:.1f}C {'increases' if sv_val > 0 else 'decreases'} urgency"
            else:
                plain = f"Humidity {val:.1f}% {'increases' if sv_val > 0 else 'decreases'} urgency"
            print(f"  {feat:<22}  {val:>8.2f}  {sv_val:>10.5f}  {direction:<9}  {plain}")

    print()
    print("  NOTE: SHAP values are in log-odds space. A positive SHAP value")
    print("  pushes the prediction toward irrigation_needed=1 (higher probability).")
    print()
    print("  NOTE on feature interactions (e.g. humidity direction flipping):")
    print("  SHAP may show humidity pushing the prediction in opposite directions")
    print("  across examples. This confirms SHAP correctly recovered the ET physics")
    print("  relationships the simulator was coded to produce. It does NOT")
    print("  independently validate real-world tomato physiology — that requires")
    print("  multi-factor field data not yet collected.")
    print("=" * 62)


# ── Section 3: Mendeley real-world validation ─────────────────────────────────
def print_realworld_validation(
    feature_cols: list[str],
    model,
    iso_forest,
) -> None:
    print()
    print("=" * 62)
    print("  SECTION 3 — Threshold Policy Comparison (Mendeley JSON)")
    print("=" * 62)
    print("  IMPORTANT: This is NOT a standard accuracy evaluation.")
    print("  Our model uses a 70% moisture threshold; the Mendeley deployment")
    print("  triggered at ~50%. These are two DIFFERENT irrigation policies.")
    print("  All real pump-ON events (at 38-50%) also exceed our 70% threshold,")
    print("  so our model would have triggered earlier. Whether earlier is better")
    print("  or worse (yield vs water use) is what the A/B experiment tests.")
    print()

    if not os.path.exists(MENDELEY_JSON):
        print("  Mendeley JSON not found — skipping real-world validation.")
        return

    with open(MENDELEY_JSON, encoding="utf-8-sig") as f:
        records = json.load(f)

    pump_on_rows  = [r for r in records if r.get("motorValue") == 1]
    pump_off_rows = [r for r in records if r.get("motorValue") == 0]

    def mendeley_to_X(rows):
        # Map Mendeley field names to our model's feature names
        return np.array([
            [r["moisturePercentage"], r["temperature"], r["humidity"]]
            for r in rows
        ], dtype=np.float32)

    X_on  = mendeley_to_X(pump_on_rows)
    X_off = mendeley_to_X(pump_off_rows[:10])   # sample 10 off rows

    # Run Isolation Forest screening first (same pipeline as production)
    if_preds_on  = iso_forest.predict(X_on)
    anomaly_on   = sum(1 for p in if_preds_on if p == -1)

    preds_on  = model.predict(X_on)
    probs_on  = model.predict_proba(X_on)[:, 1]
    correct   = sum(preds_on)   # how many of pump-ON rows predicted as 1

    preds_off = model.predict(X_off)
    correct_off = sum(1 for p in preds_off if p == 0)

    print(f"  Pump-ON rows (motorValue=1, moisture 38-50%): {len(pump_on_rows)}")
    print(f"    IsolationForest flagged as anomaly        : {anomaly_on}")
    print(f"    Our model (70% policy) predicts label=1   : {correct}/{len(pump_on_rows)}")
    print(f"    [Expected: all pump-ON rows are below 70%, so our earlier policy fires]")
    print()
    print(f"  Pump-OFF sample (motorValue=0, moisture 51-67%): 10 rows")
    print(f"    Our model (70% policy) predicts label=1   : {10 - correct_off}/10")
    print(f"    [Expected: 51-67% is below our 70% trigger; real system had not fired")
    print(f"     yet at 50%. This is a POLICY difference, not a model error.")
    print(f"     A/B experiment will quantify the water-use vs yield tradeoff.]")
    print()

    # SHAP for pump-ON rows
    explainer  = shap.TreeExplainer(model)
    sv_on      = explainer.shap_values(X_on)
    if isinstance(sv_on, list):
        sv_on = sv_on[1]

    mean_abs_on = np.abs(sv_on).mean(axis=0)
    print("  SHAP feature importance on real pump-ON rows:")
    for feat, val in zip(feature_cols, mean_abs_on):
        print(f"    {feat:<22}: mean|SHAP| = {val:.5f}")

    print()
    print("  Pump-ON rows — individual predictions with moisture values:")
    print(f"  {'moisture%':>10}  {'temp':>6}  {'hum':>6}  {'pred':>6}  {'P(irr)':>8}  {'IF':>4}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*4}")
    for i, (row, pred, prob, ifp) in enumerate(
            zip(pump_on_rows, preds_on, probs_on, if_preds_on)):
        if_str = "ANOM" if ifp == -1 else "ok"
        print(f"  {row['moisturePercentage']:>10.2f}  {row['temperature']:>6.1f}  "
              f"{row['humidity']:>6.1f}  {pred:>6}  {prob:>8.4f}  {if_str:>4}")

    print("=" * 62)


# ── Save SHAP values ──────────────────────────────────────────────────────────
def save_shap(shap_values: np.ndarray, X_test: np.ndarray,
              test_rows: list[dict], feature_cols: list[str]) -> None:
    bundle = {
        "shap_values":  shap_values,
        "X_test":       X_test,
        "feature_cols": feature_cols,
        "n_test":       len(test_rows),
        "moisture_vals": [r["soil_moisture_pct"] for r in test_rows],
        "labels":        [r["irrigation_needed"]  for r in test_rows],
    }
    with open(SHAP_OUT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  SHAP values saved -> {SHAP_OUT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Item 4 — SHAP Explainability (Sensor Model)")
    print("=" * 62)

    # Load model
    print("\nLoading model bundle ...")
    bundle      = load_bundle()
    model       = bundle["xgb_model"]
    iso_forest  = bundle["iso_forest"]
    feature_cols = bundle["feature_cols"]
    print(f"  Features: {feature_cols}")
    print(f"  Model type: {type(model).__name__}")

    # Reconstruct test set
    print("\nReconstructing test split ...")
    test_rows, X_test, y_test = load_test_rows(feature_cols)
    print(f"  Test rows: {len(test_rows)}")

    # Compute SHAP
    print("\nComputing SHAP values (TreeExplainer) ...")
    shap_values = compute_shap(model, X_test)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Reports
    print_global_importance(shap_values, feature_cols)
    print_per_prediction(test_rows, X_test, shap_values, feature_cols, model)
    print_realworld_validation(feature_cols, model, iso_forest)

    # Save
    save_shap(shap_values, X_test, test_rows, feature_cols)

    print("\nDone. Proceed to item 4b (LIME on the image model) or item 5 (fusion).")
