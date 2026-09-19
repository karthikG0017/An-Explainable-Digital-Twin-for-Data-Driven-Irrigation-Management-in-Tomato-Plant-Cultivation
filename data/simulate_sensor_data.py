"""
simulate_sensor_data.py
-----------------------
Generates a realistic multi-week sensor dataset for two tomato plants,
mimicking what an ESP32 + capacitive soil moisture sensor + DHT22 would
produce and push to Firebase.

Plant A  →  fixed-schedule irrigation (watered every 3 days regardless of need)
Plant B  →  AI-managed irrigation (watered only when soil moisture drops below 40%)

Output
------
  data/raw/plant_a_raw.csv
  data/raw/plant_b_raw.csv

Each row represents one sensor reading (every 30 minutes, matching a realistic
IoT polling interval).

Columns
-------
  timestamp          : ISO-8601 datetime string (what Firebase would store)
  plant_id           : "plant_a" or "plant_b"
  soil_moisture_pct  : 0-100 scale (100 = fully saturated / field capacity)
  temperature_c      : ambient temperature in degrees C
  humidity_pct       : relative humidity in %
  watered            : 1 if an irrigation event occurred at this reading, else 0
"""

import os
import random
import math
import csv
from datetime import datetime, timedelta

# -- Reproducibility -----------------------------------------------------------
SEED = 42
random.seed(SEED)

# -- Simulation parameters -----------------------------------------------------
START_DATE        = datetime(2024, 6, 1, 6, 0, 0)   # 06:00 on June 1
DURATION_DAYS     = 42                                # 6 weeks of data
READING_INTERVAL  = timedelta(minutes=30)             # one reading every 30 min

# Soil moisture physics
EVAPOTRANSPIRATION_PER_HOUR = 0.25   # % lost per hour under normal conditions
IRRIGATION_BOOST            = 35.0   # % added to soil moisture after watering
MOISTURE_FLOOR              = 10.0   # sensor physical minimum (never truly 0)
MOISTURE_CEILING            = 95.0   # sensor physical maximum after watering

# Plant A: fixed schedule -- water every N days
PLANT_A_WATER_INTERVAL_DAYS = 3

# Plant B: AI-managed -- water when moisture drops below this threshold
PLANT_B_TRIGGER_THRESHOLD   = 40.0  # % (deliberately low to show stress periods)

# Noise amplitudes (realistic sensor jitter for ESP32 + capacitive sensor)
MOISTURE_NOISE_STD  = 1.2   # standard deviation in %
TEMP_NOISE_STD      = 0.4   # degrees C
HUMIDITY_NOISE_STD  = 1.5   # %

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


# -- Helper: daily temperature/humidity curve ----------------------------------
def ambient_at(dt):
    """
    Returns (temperature_c, humidity_pct) for a given datetime.
    Models a realistic Indian summer day:
      - Temp peaks ~38 C at 14:00, dips to ~26 C at 04:00
      - Humidity is inversely correlated with temperature
      - Day-to-day variation simulates weather fluctuation
    """
    # Deterministic day-level "weather seed" -- same each day, different per day
    day_seed = dt.toordinal()
    day_rng  = random.Random(day_seed)

    base_temp_day = day_rng.uniform(-2.0, 3.0)   # day-to-day offset in C
    base_hum_day  = day_rng.uniform(-5.0, 5.0)   # day-to-day offset %

    hour_frac = dt.hour + dt.minute / 60.0
    # Sinusoidal: peak at 14:00 (hour 14), trough at 02:00 (hour 2)
    # sin arg: shift so that hour=14 is peak
    temp_swing = math.sin(math.pi * (hour_frac - 2) / 12)   # -1 to +1
    temp_swing = max(temp_swing, -1.0)

    temp = 32.0 + 6.0 * temp_swing + base_temp_day
    hum  = 60.0 - 20.0 * temp_swing + base_hum_day  # inverse of temp

    # Clamp to physically plausible ranges for this climate
    temp = max(24.0, min(42.0, temp))
    hum  = max(30.0, min(95.0, hum))
    return temp, hum


# -- Helper: apply Gaussian noise ----------------------------------------------
def jitter(value, std, lo, hi):
    """Add Gaussian sensor noise and clamp to valid range."""
    return max(lo, min(hi, value + random.gauss(0, std)))


# -- Core simulator ------------------------------------------------------------
def simulate_plant(plant_id, water_every_n_days, ai_trigger_pct):
    """
    Simulates sensor readings for one plant over DURATION_DAYS.

    Parameters
    ----------
    plant_id           : str, "plant_a" or "plant_b"
    water_every_n_days : int or None. Fixed-schedule day interval (Plant A).
                         Pass None to use AI mode instead.
    ai_trigger_pct     : float or None. Moisture % threshold for AI watering (Plant B).
                         Pass None to use fixed-schedule mode instead.

    Returns a list of row dicts ready for CSV writing.
    """
    rows = []
    current_dt    = START_DATE
    end_dt        = START_DATE + timedelta(days=DURATION_DAYS)

    soil_moisture = 75.0   # start near field capacity
    last_watered  = START_DATE - timedelta(days=1)

    # Track the ordinal day of the last fixed watering to avoid double-watering
    last_fixed_water_day = -999

    while current_dt < end_dt:
        # -- Evapotranspiration step ------------------------------------------
        # More moisture loss during hot midday hours, less at night
        temp_raw, hum_raw = ambient_at(current_dt)
        hour_frac   = current_dt.hour + current_dt.minute / 60.0
        et_modifier = 1.0 + 0.8 * math.sin(max(0, math.pi * (hour_frac - 6) / 12))
        et_loss     = EVAPOTRANSPIRATION_PER_HOUR * (READING_INTERVAL.seconds / 3600) * et_modifier
        soil_moisture -= et_loss
        soil_moisture  = max(MOISTURE_FLOOR, soil_moisture)

        # -- Irrigation decision -----------------------------------------------
        watered = 0

        if water_every_n_days is not None:
            # Plant A: fixed schedule -- water at 07:00 on the scheduled day
            is_water_hour = (current_dt.hour == 7 and current_dt.minute == 0)
            day_offset    = (current_dt.toordinal() - START_DATE.toordinal())
            is_water_day  = (day_offset % water_every_n_days == 0)
            not_yet_today = (current_dt.toordinal() != last_fixed_water_day)

            if is_water_day and is_water_hour and not_yet_today:
                soil_moisture        = min(MOISTURE_CEILING, soil_moisture + IRRIGATION_BOOST)
                watered              = 1
                last_watered         = current_dt
                last_fixed_water_day = current_dt.toordinal()

        else:
            # Plant B: AI-managed -- water immediately when threshold crossed
            if soil_moisture < ai_trigger_pct:
                soil_moisture = min(MOISTURE_CEILING, soil_moisture + IRRIGATION_BOOST)
                watered       = 1
                last_watered  = current_dt

        # -- Add sensor noise --------------------------------------------------
        noisy_moisture = jitter(soil_moisture, MOISTURE_NOISE_STD, MOISTURE_FLOOR, MOISTURE_CEILING)
        noisy_temp     = jitter(temp_raw,      TEMP_NOISE_STD,     15.0, 50.0)
        noisy_hum      = jitter(hum_raw,       HUMIDITY_NOISE_STD, 10.0, 100.0)

        rows.append({
            "timestamp":         current_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "plant_id":          plant_id,
            "soil_moisture_pct": round(noisy_moisture, 2),
            "temperature_c":     round(noisy_temp, 2),
            "humidity_pct":      round(noisy_hum, 2),
            "watered":           watered,
        })

        current_dt += READING_INTERVAL

    return rows


# -- Writer --------------------------------------------------------------------
def write_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["timestamp", "plant_id", "soil_moisture_pct",
                  "temperature_c", "humidity_pct", "watered"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows):,} rows -> {filepath}")


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Simulating sensor data ...\n")

    # Plant A -- fixed schedule, waters every 3 days at 07:00
    plant_a_rows = simulate_plant(
        plant_id           = "plant_a",
        water_every_n_days = PLANT_A_WATER_INTERVAL_DAYS,
        ai_trigger_pct     = None,
    )

    # Plant B -- AI-managed, waters whenever soil drops below 40%
    plant_b_rows = simulate_plant(
        plant_id           = "plant_b",
        water_every_n_days = None,
        ai_trigger_pct     = PLANT_B_TRIGGER_THRESHOLD,
    )

    write_csv(plant_a_rows, os.path.join(OUTPUT_DIR, "plant_a_raw.csv"))
    write_csv(plant_b_rows, os.path.join(OUTPUT_DIR, "plant_b_raw.csv"))

    # -- Quick sanity summary -------------------------------------------------
    print("\n-- Plant A (fixed schedule) ----------------------------------------")
    a_moisture = [r["soil_moisture_pct"] for r in plant_a_rows]
    a_watered  = sum(r["watered"] for r in plant_a_rows)
    print(f"  Irrigation events : {a_watered}")
    print(f"  Soil moisture     : min={min(a_moisture):.1f}%  "
          f"max={max(a_moisture):.1f}%  avg={sum(a_moisture)/len(a_moisture):.1f}%")

    print("\n-- Plant B (AI-managed) --------------------------------------------")
    b_moisture = [r["soil_moisture_pct"] for r in plant_b_rows]
    b_watered  = sum(r["watered"] for r in plant_b_rows)
    print(f"  Irrigation events : {b_watered}")
    print(f"  Soil moisture     : min={min(b_moisture):.1f}%  "
          f"max={max(b_moisture):.1f}%  avg={sum(b_moisture)/len(b_moisture):.1f}%")

    print("\nDone. Feed these CSVs into the label-engineering script (item 2).")
