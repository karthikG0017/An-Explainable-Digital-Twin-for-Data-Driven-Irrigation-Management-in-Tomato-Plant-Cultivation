"""
label_engineering.py
--------------------
Reads the raw simulated sensor CSVs from Item 1 and engineers two new columns:

  irrigation_needed  (int, 0 or 1)
      1  if soil_moisture_pct < 70%   → plant needs water
      0  if soil_moisture_pct >= 70%  → plant is adequately hydrated

      Threshold source: published crop-water-stress research stating that
      70–100% field capacity sustains healthy tomato yield; below 70%
      yield starts to decline, and below 40–50% decline is severe.

  stress_level  (str)
      "healthy"         if soil_moisture_pct >= 70%
      "moderate_stress" if 50% <= soil_moisture_pct < 70%
      "high_stress"     if soil_moisture_pct < 50%

Input  : data/raw/plant_a_raw.csv, data/raw/plant_b_raw.csv
Output : data/processed/plant_a_labeled.csv, data/processed/plant_b_labeled.csv
"""

import os
import csv
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INPUT_FILES = {
    "plant_a": os.path.join(RAW_DIR, "plant_a_raw.csv"),
    "plant_b": os.path.join(RAW_DIR, "plant_b_raw.csv"),
}
OUTPUT_FILES = {
    "plant_a": os.path.join(PROCESSED_DIR, "plant_a_labeled.csv"),
    "plant_b": os.path.join(PROCESSED_DIR, "plant_b_labeled.csv"),
}

# ── Thresholds (match project spec exactly) ────────────────────────────────────
IRRIGATION_THRESHOLD   = 70.0   # below this → irrigation_needed = 1
HIGH_STRESS_THRESHOLD  = 50.0   # below this → high_stress (subset of above)


# ── Label functions ────────────────────────────────────────────────────────────
def get_irrigation_needed(moisture: float) -> int:
    """Binary label: 1 if plant needs irrigation, 0 if adequately hydrated."""
    return 1 if moisture < IRRIGATION_THRESHOLD else 0


def get_stress_level(moisture: float) -> str:
    """Three-class stress category used for explainability and reporting."""
    if moisture >= IRRIGATION_THRESHOLD:
        return "healthy"
    elif moisture >= HIGH_STRESS_THRESHOLD:
        return "moderate_stress"
    else:
        return "high_stress"


# ── Core processing function ───────────────────────────────────────────────────
def label_plant(plant_id: str) -> list[dict]:
    """
    Reads raw CSV for one plant, adds label columns, returns list of row dicts.
    """
    input_path = INPUT_FILES[plant_id]

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}\n"
            "Run simulate_sensor_data.py first."
        )

    labeled_rows = []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            moisture = float(row["soil_moisture_pct"])

            # Add the two new columns -- all original columns are preserved
            row["irrigation_needed"] = get_irrigation_needed(moisture)
            row["stress_level"]      = get_stress_level(moisture)
            labeled_rows.append(row)

    return labeled_rows


# ── Writer ─────────────────────────────────────────────────────────────────────
def write_labeled_csv(rows: list[dict], plant_id: str) -> None:
    output_path = OUTPUT_FILES[plant_id]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Preserve original column order, append new columns at the end
    fieldnames = [
        "timestamp", "plant_id", "soil_moisture_pct",
        "temperature_c", "humidity_pct", "watered",
        "irrigation_needed", "stress_level",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows):,} rows -> {output_path}")


# ── Summary printer ────────────────────────────────────────────────────────────
def print_summary(plant_id: str, rows: list[dict]) -> None:
    total = len(rows)

    # stress_level distribution
    stress_counts = Counter(r["stress_level"] for r in rows)
    irr_count     = sum(int(r["irrigation_needed"]) for r in rows)

    # Average moisture per stress level (useful for sanity-checking thresholds)
    moisture_by_stress: dict[str, list[float]] = {
        "healthy":         [],
        "moderate_stress": [],
        "high_stress":     [],
    }
    for r in rows:
        moisture_by_stress[r["stress_level"]].append(float(r["soil_moisture_pct"]))

    label_order = ["healthy", "moderate_stress", "high_stress"]

    print(f"\n{'=' * 58}")
    print(f"  {plant_id.upper()}  —  Label Distribution")
    print(f"{'=' * 58}")
    print(f"  Total rows         : {total:,}")
    print(f"  irrigation_needed=1: {irr_count:,}  ({irr_count/total*100:.1f}%)")
    print(f"  irrigation_needed=0: {total - irr_count:,}  ({(total-irr_count)/total*100:.1f}%)")
    print()
    print(f"  {'Stress Level':<20} {'Count':>6}  {'%':>6}  {'Avg Moisture':>14}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*14}")
    for level in label_order:
        count  = stress_counts.get(level, 0)
        pct    = count / total * 100
        bucket = moisture_by_stress[level]
        avg_m  = sum(bucket) / len(bucket) if bucket else 0.0
        print(f"  {level:<20} {count:>6,}  {pct:>6.1f}%  {avg_m:>13.1f}%")
    print(f"{'=' * 58}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Engineering labels ...\n")

    for plant_id in ("plant_a", "plant_b"):
        rows = label_plant(plant_id)
        write_labeled_csv(rows, plant_id)
        print_summary(plant_id, rows)

    print("\nDone. Feed data/processed/*_labeled.csv into the sensor model (item 3).")
