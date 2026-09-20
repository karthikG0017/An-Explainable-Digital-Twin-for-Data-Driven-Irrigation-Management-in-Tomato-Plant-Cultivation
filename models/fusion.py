"""
fusion.py
---------
Item 5: Rule-based fusion layer combining the XGBoost sensor model and the
EfficientNetB0 image model into a single irrigation + disease decision.

DESIGN PHILOSOPHY
=================
This is a RULE-BASED fusion layer, not a learned combiner. Reasons:
  1. We have two heterogeneous input types (tabular sensor data + images)
     with very different confidence distributions.
  2. A learned combiner would require paired (sensor+image) training samples
     from the SAME plant at the SAME timestamp, which our dataset does not
     provide — sensor data is continuous; images are episodic.
  3. Rule-based logic is fully auditable and explainable, which is a core
     thesis requirement.

FUSION RULES (ordered by priority)
====================================
Rule 1 — Anomaly gate:
  If the sensor reading is flagged as anomalous by the Isolation Forest,
  the sensor model output is unreliable. Return SENSOR_ANOMALY status;
  suppress sensor-based irrigation trigger.

Rule 2 — Image confidence gate:
  If the image model's top-1 probability < IMAGE_CONF_THRESHOLD (0.70),
  the disease classification is marked LOW_CONFIDENCE and downweighted.

Rule 3 — Known-confusable-pair flag:
  If the predicted class is Target_Spot or Spider_mites AND the second-most-
  likely probability is the other member of the pair AND the gap between
  them is < CONFUSABLE_GAP_THRESHOLD (0.35), flag as UNCERTAIN_PAIR.
  Rationale: empirical analysis on the test set showed 57% of Target_Spot
  misclassifications as Spider_mites fall below a 0.35 gap, while only
  ~20% of correct Target_Spot predictions are affected (see evaluate_image_model.py
  and gap calibration analysis in the project log).

Rule 4 — Irrigation decision:
  Primary: sensor model prediction (XGBoost on moisture/temp/humidity).
  Disease context (image model) does NOT override irrigation trigger —
  it enriches the alert with disease context only.
  Exception: if sensor model says irrigation_needed=0 BUT image model
  detects a disease associated with root stress or wilting (Late_blight,
  Bacterial_spot), emit a DISEASE_STRESS_WATCH advisory (not a trigger).

Rule 5 — Final status codes:
  IRRIGATE          : sensor model says irrigate, all signals consistent
  IRRIGATE_WITH_ALERT: irrigate + disease detected at high confidence
  HOLD              : sensor model says do not irrigate
  HOLD_WITH_ALERT   : hold + disease detected at high confidence
  DISEASE_STRESS_WATCH: hold + disease detected suggesting future stress
  SENSOR_ANOMALY    : sensor reading rejected by Isolation Forest
  UNCERTAIN         : low confidence or confusable pair — report but do not act

IMPORTANT LIMITATIONS (document in thesis)
==========================================
  - This fusion layer makes no assumptions about which plant a sensor
    reading or image comes from — it processes (sensor_row, image_probs)
    as an independent paired observation.
  - The 70% confidence threshold and 0.35 gap threshold were calibrated
    on the PlantVillage intra-dataset test split. Real-field confidence
    distributions may differ (domain gap).
  - The known-confusable-pair flag does not prevent all Target_Spot/Spider_mites
    errors. Combined analysis (confidence < 0.70 OR pair gap < 0.35) catches
    24/35 misclassifications (68.6%). The remaining 11/35 (31.4%) are high-
    confidence wrong predictions (top-1 prob 0.73-0.95) that neither flag
    detects. This equals 11/138 = 8.0% of the Target_Spot test set — the
    irreducible undetected error rate at current model capacity. Cite this
    figure consistently in the thesis; do not use the earlier approximation.
  - Disease detection is treated as ADVISORY, not a direct irrigation override.
    Whether disease-associated stress affects irrigation needs in practice
    is a field question for the A/B experiment.

Input  : sensor_reading (dict: soil_moisture_pct, temperature_c, humidity_pct)
         image_probs (dict: class_name -> probability)  OR  None
Output : FusionResult (dataclass with all fields documented below)
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR              = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH            = os.path.join(ROOT_DIR, "xgboost_irrigation.pkl")
CM_PATH               = os.path.join(ROOT_DIR, "confusion_matrix.json")
TREATMENT_LOOKUP_PATH = os.path.join(os.path.dirname(ROOT_DIR),
                                     "data", "disease_treatment_lookup.json")

# Thresholds (empirically calibrated — see gap_calibration analysis)
IMAGE_CONF_THRESHOLD   = 0.70   # below this: low-confidence image prediction
CONFUSABLE_GAP_MARGIN  = 0.35   # Target_Spot/Spider_mites gap threshold
SENSOR_IRRIG_THRESHOLD = 0.50   # XGBoost P(irrigation_needed=1) for positive call

# Classes where literature associates the disease with conditions that may
# affect moisture dynamics over time (canopy damage, stomatal disruption).
# This is a watch signal only — NOT a confirmed causal mechanism, and NOT
# validated by this project's sensor+image paired field data.
# See Rule 4 and the DISEASE_STRESS_WATCH alert wording below.
STRESS_ASSOCIATED_DISEASES = {
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
}

# The known-confusable pair (empirically identified)
CONFUSABLE_PAIR = frozenset([
    "Tomato__Target_Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
])


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class FusionResult:
    # Core decision
    status: str                          # see Rule 5 codes above
    irrigate: bool                       # final actionable irrigation trigger

    # Sensor model outputs
    sensor_irrigation_needed: int        # XGBoost binary prediction (0 or 1)
    sensor_probability: float            # P(irrigation_needed=1)
    sensor_anomaly: bool                 # True if Isolation Forest rejected reading
    soil_moisture_pct: float
    temperature_c: float
    humidity_pct: float

    # Image model outputs (None if no image provided)
    disease_class: Optional[str]         # top predicted class name
    disease_confidence: float            # top-1 probability
    disease_low_confidence: bool         # True if below IMAGE_CONF_THRESHOLD
    known_confusable_pair: bool          # True if Target_Spot/Spider_mites ambiguity
    top2_gap: Optional[float]            # P(top1) - P(top2), or None
    disease_stress_watch: bool           # True if stress-associated disease detected

    # Treatment recommendation (populated whenever disease_class is not None)
    # Contains the full ICAR-sourced entry: display_name, pathogen, severity,
    # urgency, immediate_actions, chemical_treatments, biological_treatments,
    # irrigation_note. Always self-contained so the dashboard needs no
    # secondary lookup — one API call gives the complete decision + advice.
    treatment: dict

    # Explanation fields (for SHAP/LIME integration in Flask API)
    sensor_explanation: dict             # placeholder for SHAP values at API layer
    image_explanation: dict             # placeholder for LIME superpixel mask

    # Audit
    alerts: list[str] = field(default_factory=list)  # human-readable alert messages


# ── Load model bundle (once at import time) ────────────────────────────────────
_MODEL_BUNDLE: Optional[dict] = None

def _get_bundle() -> dict:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Sensor model not found: {MODEL_PATH}\n"
                "Run models/train_sensor_model.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _MODEL_BUNDLE = pickle.load(f)
    return _MODEL_BUNDLE


# ── Load treatment lookup (once at import time) ────────────────────────────────
_TREATMENT_LOOKUP: Optional[dict] = None

def _get_treatment(disease_class: Optional[str]) -> dict:
    """
    Returns the full ICAR treatment entry for a given disease class name,
    or an empty dict if no image was provided or class not found.
    Always included in FusionResult so the dashboard never has to look it up
    separately — every API response is self-contained.
    """
    global _TREATMENT_LOOKUP
    if disease_class is None:
        return {}
    if _TREATMENT_LOOKUP is None:
        if not os.path.exists(TREATMENT_LOOKUP_PATH):
            return {"error": f"Treatment lookup not found: {TREATMENT_LOOKUP_PATH}"}
        with open(TREATMENT_LOOKUP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Strip the _meta key — not a disease entry
        _TREATMENT_LOOKUP = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _TREATMENT_LOOKUP.get(disease_class, {
        "display_name": disease_class,
        "error": "Class not found in treatment lookup — update disease_treatment_lookup.json"
    })


# ── Sensor model inference ────────────────────────────────────────────────────
def run_sensor_model(
    soil_moisture_pct: float,
    temperature_c: float,
    humidity_pct: float,
) -> tuple[int, float, bool]:
    """
    Returns (prediction, probability, is_anomaly).
    Applies Isolation Forest first, then XGBoost — same pipeline as training.
    """
    bundle = _get_bundle()
    xgb       = bundle["xgb_model"]
    iso       = bundle["iso_forest"]
    feat_cols = bundle["feature_cols"]   # ['soil_moisture_pct','temperature_c','humidity_pct']

    X = np.array([[soil_moisture_pct, temperature_c, humidity_pct]], dtype=np.float32)

    # Isolation Forest: -1 = anomaly, 1 = normal
    is_anomaly = iso.predict(X)[0] == -1

    pred  = int(xgb.predict(X)[0])
    prob  = float(xgb.predict_proba(X)[0][1])   # P(irrigation_needed=1)

    return pred, prob, is_anomaly


# ── Image model analysis ──────────────────────────────────────────────────────
def analyze_image_probs(
    image_probs: Optional[dict[str, float]],
) -> tuple[Optional[str], float, bool, bool, Optional[float]]:
    """
    Returns:
      disease_class        : top-1 class name (or None)
      disease_confidence   : top-1 probability
      low_confidence       : True if below IMAGE_CONF_THRESHOLD
      known_confusable     : True if Target_Spot/Spider_mites pair ambiguity
      top2_gap             : P(top1) - P(top2), or None
    """
    if not image_probs:
        return None, 0.0, False, False, None

    sorted_probs = sorted(image_probs.items(), key=lambda x: -x[1])
    top1_class, top1_prob = sorted_probs[0]
    top2_class, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)

    gap = top1_prob - top2_prob if top2_class else None

    low_conf = top1_prob < IMAGE_CONF_THRESHOLD

    # Known confusable pair check:
    # Triggers only if BOTH top-1 AND top-2 are members of the pair AND gap < margin.
    known_confusable = (
        top1_class in CONFUSABLE_PAIR
        and top2_class in CONFUSABLE_PAIR
        and gap is not None
        and gap < CONFUSABLE_GAP_MARGIN
    )

    return top1_class, top1_prob, low_conf, known_confusable, gap


# ── Core fusion function ──────────────────────────────────────────────────────
def fuse(
    soil_moisture_pct: float,
    temperature_c: float,
    humidity_pct: float,
    image_probs: Optional[dict[str, float]] = None,
    sensor_shap_values: Optional[dict] = None,
    image_lime_mask: Optional[dict] = None,
) -> FusionResult:
    """
    Main entry point. Call with sensor readings and optionally image probabilities.

    Parameters
    ----------
    soil_moisture_pct : float   0-100 scale
    temperature_c     : float   Celsius
    humidity_pct      : float   0-100 scale
    image_probs       : dict    {class_name: probability} from EfficientNetB0 softmax
                                Pass None if no image is available for this reading.
    sensor_shap_values: dict    Pre-computed SHAP values for this reading (from API layer)
    image_lime_mask   : dict    LIME superpixel mask for this image (from API layer)

    Returns
    -------
    FusionResult dataclass (use asdict() to serialise for JSON API response)
    """
    alerts = []

    # ── Rule 1: Sensor model ─────────────────────────────────────────────────
    sensor_pred, sensor_prob, is_anomaly = run_sensor_model(
        soil_moisture_pct, temperature_c, humidity_pct
    )

    if is_anomaly:
        alerts.append(
            "SENSOR_ANOMALY: Reading flagged by Isolation Forest as out-of-distribution. "
            "Check sensor hardware. Irrigation decision suppressed."
        )

    # ── Rule 2 & 3: Image model ───────────────────────────────────────────────
    disease_class, disease_conf, low_conf, known_confusable, top2_gap = analyze_image_probs(
        image_probs
    )

    if low_conf and disease_class:
        alerts.append(
            f"IMAGE_LOW_CONFIDENCE: Disease prediction '{disease_class}' has confidence "
            f"{disease_conf:.1%} (below {IMAGE_CONF_THRESHOLD:.0%} threshold). "
            "Treat as uncertain — do not report to grower as definitive."
        )

    if known_confusable:
        alerts.append(
            f"UNCERTAIN_PAIR: Model cannot reliably distinguish Target_Spot from "
            f"Spider_mites (top-2 gap={top2_gap:.3f} < {CONFUSABLE_GAP_MARGIN}). "
            "This is a known failure mode (empirically characterised in evaluation). "
            "Report both candidates to grower; recommend visual inspection."
        )

    # ── Rule 4: Disease-stress watch ─────────────────────────────────────────
    # Only fires if confident, not confusable, and specific stress-linked disease
    stress_watch = (
        disease_class in STRESS_ASSOCIATED_DISEASES
        and not low_conf
        and not known_confusable
        and sensor_pred == 0   # sensor says no irrigation needed right now
    )
    if stress_watch:
        alerts.append(
            f"DISEASE_STRESS_WATCH: {disease_class} detected at {disease_conf:.1%} confidence. "
            "NOTE: Sensor moisture reading is within normal range — irrigation is NOT currently "
            "triggered. These are two independent findings. "
            "Literature associates this disease with conditions that can accelerate moisture "
            "depletion over time (canopy damage, stomatal disruption), but this project has "
            "not validated that link with field data. Monitor moisture closely over next 6-12h "
            "and act on sensor readings, not on this alert alone."
        )

    # ── Rule 5: Determine final status ───────────────────────────────────────
    if is_anomaly:
        status   = "SENSOR_ANOMALY"
        irrigate = False

    elif known_confusable or (low_conf and sensor_pred == 0):
        # If sensor says hold AND image is uncertain, report as uncertain
        status   = "UNCERTAIN"
        irrigate = False

    elif sensor_pred == 1:
        # Sensor says irrigate
        if disease_class and not low_conf and not known_confusable:
            status = "IRRIGATE_WITH_ALERT"
        else:
            status = "IRRIGATE"
        irrigate = True

    else:
        # Sensor says hold
        if stress_watch:
            status   = "DISEASE_STRESS_WATCH"
            irrigate = False
        elif disease_class and not low_conf and not known_confusable:
            status   = "HOLD_WITH_ALERT"
            irrigate = False
        else:
            status   = "HOLD"
            irrigate = False

    return FusionResult(
        status                   = status,
        irrigate                 = irrigate,
        sensor_irrigation_needed = sensor_pred,
        sensor_probability       = round(sensor_prob, 4),
        sensor_anomaly           = is_anomaly,
        soil_moisture_pct        = soil_moisture_pct,
        temperature_c            = temperature_c,
        humidity_pct             = humidity_pct,
        disease_class            = disease_class,
        disease_confidence       = round(disease_conf, 4),
        disease_low_confidence   = low_conf,
        known_confusable_pair    = known_confusable,
        top2_gap                 = round(top2_gap, 4) if top2_gap is not None else None,
        disease_stress_watch     = stress_watch,
        treatment                = _get_treatment(disease_class),
        sensor_explanation       = sensor_shap_values or {},
        image_explanation        = image_lime_mask or {},
        alerts                   = alerts,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 68)
    print("  Item 5 — Fusion Logic Self-Test")
    print("=" * 68)

    test_cases = [
        {
            "name": "Healthy, no image",
            "moisture": 82.0, "temp": 28.5, "hum": 65.0,
            "image_probs": None,
        },
        {
            "name": "Stressed soil (irrigate), clean disease signal",
            "moisture": 55.0, "temp": 31.0, "hum": 78.0,
            "image_probs": {
                "Tomato_Bacterial_spot": 0.87,
                "Tomato_Early_blight":  0.06,
                "Tomato_healthy":       0.03,
                "Tomato_Late_blight":   0.02,
                "Tomato__Target_Spot":  0.01,
                "Tomato_Septoria_leaf_spot": 0.005,
                "Tomato_Spider_mites_Two_spotted_spider_mite": 0.002,
                "Tomato_Leaf_Mold": 0.001,
                "Tomato__Tomato_YellowLeaf__Curl_Virus": 0.001,
                "Tomato__Tomato_mosaic_virus": 0.001,
            },
        },
        {
            "name": "Target_Spot/Spider_mites confusable pair",
            "moisture": 75.0, "temp": 29.0, "hum": 70.0,
            "image_probs": {
                "Tomato__Target_Spot": 0.48,
                "Tomato_Spider_mites_Two_spotted_spider_mite": 0.44,
                "Tomato_Bacterial_spot": 0.04,
                "Tomato_Early_blight": 0.02,
                "Tomato_Late_blight": 0.01,
                "Tomato_healthy": 0.005,
                "Tomato_Septoria_leaf_spot": 0.002,
                "Tomato_Leaf_Mold": 0.001,
                "Tomato__Tomato_YellowLeaf__Curl_Virus": 0.001,
                "Tomato__Tomato_mosaic_virus": 0.001,
            },
        },
        {
            "name": "Disease stress watch (Late blight, sensor OK)",
            "moisture": 76.0, "temp": 30.0, "hum": 80.0,
            "image_probs": {
                "Tomato_Late_blight": 0.91,
                "Tomato_Bacterial_spot": 0.04,
                "Tomato_Early_blight": 0.02,
                "Tomato__Target_Spot": 0.01,
                "Tomato_healthy": 0.005,
                "Tomato_Septoria_leaf_spot": 0.003,
                "Tomato_Spider_mites_Two_spotted_spider_mite": 0.002,
                "Tomato_Leaf_Mold": 0.001,
                "Tomato__Tomato_YellowLeaf__Curl_Virus": 0.001,
                "Tomato__Tomato_mosaic_virus": 0.001,
            },
        },
        {
            "name": "Low-confidence image, stressed soil",
            "moisture": 58.0, "temp": 33.0, "hum": 60.0,
            "image_probs": {
                "Tomato_Early_blight": 0.45,
                "Tomato__Target_Spot": 0.30,
                "Tomato_Bacterial_spot": 0.15,
                "Tomato_healthy": 0.05,
                "Tomato_Late_blight": 0.02,
                "Tomato_Septoria_leaf_spot": 0.01,
                "Tomato_Spider_mites_Two_spotted_spider_mite": 0.005,
                "Tomato_Leaf_Mold": 0.003,
                "Tomato__Tomato_YellowLeaf__Curl_Virus": 0.001,
                "Tomato__Tomato_mosaic_virus": 0.001,
            },
        },
    ]

    for tc in test_cases:
        result = fuse(
            soil_moisture_pct = tc["moisture"],
            temperature_c     = tc["temp"],
            humidity_pct      = tc["hum"],
            image_probs       = tc["image_probs"],
        )
        print(f"\n  [{tc['name']}]")
        print(f"  Input : moisture={tc['moisture']}%  temp={tc['temp']}C  hum={tc['hum']}%")
        print(f"  STATUS: {result.status}")
        print(f"  IRRIGATE: {result.irrigate}")
        print(f"  Sensor P(irrigate)={result.sensor_probability:.4f}  anomaly={result.sensor_anomaly}")
        if result.disease_class:
            print(f"  Disease: {result.disease_class}  conf={result.disease_confidence:.4f}  "
                  f"low_conf={result.disease_low_confidence}  confusable={result.known_confusable_pair}")
        if result.alerts:
            for a in result.alerts:
                print(f"  ALERT: {a[:100]}...")
        print(f"  {'-'*64}")

    print("\n  Self-test complete.")
    print("  Proceed to item 6 (Flask API) to expose these via HTTP endpoints.")
