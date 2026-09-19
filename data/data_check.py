"""
data_check.py
-------------
Pre-training diagnostic. Answers three questions before the XGBoost model
is built:

  1. After dropping plant_id, does the combined dataset still have real
     feature variation — or does Plant A just add dead weight?
  2. What are the correlations between soil_moisture_pct and the other
     features (temperature_c, humidity_pct)?
  3. Is the class imbalance quantified so we can set scale_pos_weight
     correctly in item 3?

Run this script once, review the output, then proceed to item 3.

Input  : data/processed/plant_a_labeled.csv
         data/processed/plant_b_labeled.csv
Output : printed report only (no files written)
"""

import os
import csv
import math
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

FILES = {
    "plant_a": os.path.join(PROCESSED_DIR, "plant_a_labeled.csv"),
    "plant_b": os.path.join(PROCESSED_DIR, "plant_b_labeled.csv"),
}

# Features the model will actually see (plant_id intentionally excluded)
MODEL_FEATURES = ["soil_moisture_pct", "temperature_c", "humidity_pct"]
LABEL_COL      = "irrigation_needed"


# ── Load & combine ─────────────────────────────────────────────────────────────
def load_combined() -> list[dict]:
    """Load both labeled CSVs, convert numeric columns, drop plant_id."""
    all_rows = []
    for plant_id, path in FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Labeled CSV not found: {path}\n"
                "Run label_engineering.py first."
            )
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # Cast numeric columns — plant_id column is read but never used
                all_rows.append({
                    "source":            plant_id,          # kept for reporting only
                    "soil_moisture_pct": float(row["soil_moisture_pct"]),
                    "temperature_c":     float(row["temperature_c"]),
                    "humidity_pct":      float(row["humidity_pct"]),
                    "irrigation_needed": int(row["irrigation_needed"]),
                    "stress_level":      row["stress_level"],
                })
    return all_rows


# ── Statistics helpers ─────────────────────────────────────────────────────────
def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)

def std(vals: list[float]) -> float:
    m = mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)

def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    mx, my = mean(xs), mean(ys)
    sx, sy = std(xs), std(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


# ── Section 1: Per-source contribution ────────────────────────────────────────
def check_per_source_variation(rows: list[dict]) -> None:
    print("=" * 62)
    print("  SECTION 1 — Per-plant feature ranges (before combining)")
    print("=" * 62)
    print(f"  {'Source':<10} {'Feature':<22} {'Min':>7} {'Max':>7} {'Std':>7} {'Label=1%':>9}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")

    for source in ("plant_a", "plant_b"):
        src_rows = [r for r in rows if r["source"] == source]
        n_stress = sum(r["irrigation_needed"] for r in src_rows)
        label_pct = n_stress / len(src_rows) * 100

        for feat in MODEL_FEATURES:
            vals = [r[feat] for r in src_rows]
            print(f"  {source:<10}  {feat:<22}  {min(vals):>7.2f}  {max(vals):>7.2f}  "
                  f"{std(vals):>7.2f}  {label_pct if feat == MODEL_FEATURES[0] else '':>9}")

    print()
    print("  KEY QUESTION: Does Plant A add real feature variation?")
    print("  Plant A has label=1 on 0% of rows, but its moisture/temp/humidity")
    print("  readings span a different range than Plant B's stressed periods.")
    print("  The model should learn: HIGH moisture -> label=0 (regardless of source).")


# ── Section 2: Correlation matrix ─────────────────────────────────────────────
def check_correlations(rows: list[dict]) -> None:
    print()
    print("=" * 62)
    print("  SECTION 2 — Pearson correlation (combined dataset, n=%d)" % len(rows))
    print("=" * 62)
    print("  Correlations with soil_moisture_pct:")
    print()

    moisture = [r["soil_moisture_pct"] for r in rows]

    # Correlate moisture against each other feature + the label
    targets = {
        "temperature_c":     [r["temperature_c"]     for r in rows],
        "humidity_pct":      [r["humidity_pct"]      for r in rows],
        "irrigation_needed": [float(r["irrigation_needed"]) for r in rows],
    }

    for name, vals in targets.items():
        r = pearson(moisture, vals)
        bar_len = int(abs(r) * 30)
        direction = "+" if r > 0 else "-"
        bar = direction * bar_len
        print(f"  {name:<22}  r = {r:+.4f}  |{bar:<30}|")

    print()
    print("  INTERPRETATION:")
    print("  - moisture vs irrigation_needed should be strongly NEGATIVE (r ~ -1)")
    print("    meaning low moisture => label=1. If this is weak, labels are wrong.")
    print("  - moisture vs temperature should be weakly negative (hot days dry soil)")
    print("  - moisture vs humidity should be weakly positive (humid days => less ET)")


# ── Section 3: 10-row sample (mixed sources, mixed labels) ─────────────────────
def check_sample_rows(rows: list[dict]) -> None:
    print()
    print("=" * 62)
    print("  SECTION 3 — 10-row sample (5 label=0, 5 label=1, mixed sources)")
    print("=" * 62)
    print(f"  {'Source':<10} {'Moisture%':>10} {'Temp°C':>8} {'Hum%':>7} {'Label':>7} {'Stress':<18}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*18}")

    # Pick 5 label=0 rows and 5 label=1 rows deterministically (every Nth row)
    label0 = [r for r in rows if r["irrigation_needed"] == 0]
    label1 = [r for r in rows if r["irrigation_needed"] == 1]

    step0 = max(1, len(label0) // 5)
    step1 = max(1, len(label1) // 5)
    sample = [label0[i * step0] for i in range(5)] + [label1[i * step1] for i in range(5)]

    for r in sample:
        print(f"  {r['source']:<10}  {r['soil_moisture_pct']:>10.2f}  "
              f"{r['temperature_c']:>8.2f}  {r['humidity_pct']:>7.2f}  "
              f"{r['irrigation_needed']:>7}  {r['stress_level']:<18}")


# ── Section 4: Class imbalance + scale_pos_weight ────────────────────────────
def check_class_imbalance(rows: list[dict]) -> None:
    print()
    print("=" * 62)
    print("  SECTION 4 — Class imbalance & scale_pos_weight recommendation")
    print("=" * 62)

    n_total = len(rows)
    n_pos   = sum(r["irrigation_needed"] for r in rows)  # label = 1
    n_neg   = n_total - n_pos                             # label = 0
    ratio   = n_neg / n_pos if n_pos > 0 else float("inf")

    print(f"  Total rows              : {n_total:,}")
    print(f"  label=0 (not needed)    : {n_neg:,}  ({n_neg/n_total*100:.1f}%)")
    print(f"  label=1 (needed)        : {n_pos:,}  ({n_pos/n_total*100:.1f}%)")
    print()
    print(f"  Recommended scale_pos_weight = n_neg / n_pos = {ratio:.2f}")
    print()
    print("  This tells XGBoost to penalise missed irrigations (false negatives)")
    print("  more heavily than false alarms — correct for this use case, since")
    print("  failing to irrigate when needed causes crop damage.")
    print("=" * 62)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading combined dataset (plant_id dropped as model feature) ...\n")
    rows = load_combined()
    print(f"  Loaded {len(rows):,} rows total ({len(rows)//2:,} per plant)\n")

    check_per_source_variation(rows)
    check_correlations(rows)
    check_sample_rows(rows)
    check_class_imbalance(rows)

    print("\nReview the output above before proceeding to item 3 (model training).")
