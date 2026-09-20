"""
app.py
------
Item 6: Flask REST API for the Explainable Digital Twin irrigation system.

ENDPOINTS
=========

  GET  /health
  GET  /api/classes
  GET  /api/confusion_matrix
  POST /api/predict/sensor
  POST /api/predict/full

See inline docstrings for exact request/response contracts.

ARCHITECTURE NOTE
=================
This API is a thin routing layer. All decision logic lives in:
  models/fusion.py          (rule-based fusion + treatment lookup)
  backend/image_inference.py (EfficientNetB0 inference, lazy-loaded)

The API never duplicates any model logic — it only:
  1. Validates and parses inputs
  2. Calls the appropriate inference functions
  3. Computes SHAP values for sensor readings (fast, per-request)
  4. Serialises FusionResult to JSON

SHAP vs LIME
============
SHAP (sensor): computed per-request using TreeExplainer — fast (<10ms).
LIME (image) : NOT computed per-request — too slow (minutes per image).
               The `image_explanation` field is returned as {} unless the
               client explicitly requests LIME via a separate endpoint
               (not in scope for item 6).
"""

from __future__ import annotations

import os
import sys
import json
import traceback
import io
from dataclasses import asdict

import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from PIL import Image

# ── Project root on sys.path so relative imports resolve ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from models.fusion import fuse, FusionResult, _get_bundle   # noqa: E402
import backend.image_inference as img_inf                    # noqa: E402

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow cross-origin requests from React frontend on localhost

# ── SHAP explainer (sensor model, loaded once) ────────────────────────────────
_shap_explainer = None

def _get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        import shap
        bundle = _get_bundle()
        _shap_explainer = shap.TreeExplainer(bundle["xgb_model"])
    return _shap_explainer


def _compute_shap(soil_moisture_pct: float,
                  temperature_c: float,
                  humidity_pct: float) -> dict:
    """
    Returns SHAP values for a single sensor reading as a dict.
    Keys: feature names. Values: SHAP contribution (positive = pushes toward irrigation).

    NOTE: SHAP values explain the XGBoost model's raw prediction for
    irrigation_needed=1. They recover the simulator's physics relationships
    (moisture is the dominant driver). Documented as such — not independent
    validation of real-world plant physiology.
    """
    try:
        explainer = _get_shap_explainer()
        bundle    = _get_bundle()
        feat_cols = bundle["feature_cols"]
        X         = np.array([[soil_moisture_pct, temperature_c, humidity_pct]],
                             dtype=np.float32)
        sv = explainer.shap_values(X)
        # TreeExplainer for binary XGBoost returns array of shape (1, n_features)
        # for the positive class
        if isinstance(sv, list):
            sv = sv[1]   # index 1 = P(irrigation_needed=1)
        # Cast to plain Python float — numpy scalars are not JSON-serialisable
        return {feat: float(val) for feat, val in zip(feat_cols, sv[0])}
    except Exception as e:
        return {"error": str(e)}


# ── JSON-safe conversion — Flask 3 uses DefaultJSONProvider which doesn't ─────
# support subclassing json.encoder. Convert numpy types recursively instead.
def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Input validation helper ───────────────────────────────────────────────────
def _parse_sensor_fields(source) -> tuple[float, float, float]:
    """
    Parse soil_moisture_pct, temperature_c, humidity_pct from either
    a JSON body (dict) or a multipart form (ImmutableMultiDict).
    Raises ValueError with a descriptive message on missing/invalid fields.
    """
    def _get(key):
        val = source.get(key)
        if val is None:
            raise ValueError(f"Missing required field: '{key}'")
        try:
            return float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{key}' must be a number, got: {val!r}")

    moisture = _get("soil_moisture_pct")
    temp     = _get("temperature_c")
    hum      = _get("humidity_pct")

    if not (0.0 <= moisture <= 100.0):
        raise ValueError(f"soil_moisture_pct must be 0-100, got {moisture}")
    if not (-10.0 <= temp <= 60.0):
        raise ValueError(f"temperature_c must be -10 to 60, got {temp}")
    if not (0.0 <= hum <= 100.0):
        raise ValueError(f"humidity_pct must be 0-100, got {hum}")

    return moisture, temp, hum


# ── Error handler ─────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "detail": str(e)}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Health check
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    """
    GET /health

    Returns
    -------
    200 OK
    {
      "status": "ok",
      "sensor_model_loaded": bool,
      "image_model_loaded":  bool
    }

    Purpose: ping before making prediction requests.
    Sensor model is always loaded at startup. Image model is lazy-loaded
    on the first /predict/full call.
    """
    sensor_ready = True
    try:
        _get_bundle()
    except Exception:
        sensor_ready = False

    image_ready = img_inf._model is not None

    return jsonify({
        "status": "ok" if sensor_ready else "degraded",
        "sensor_model_loaded": sensor_ready,
        "image_model_loaded":  image_ready,
    })


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Class names
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/api/classes", methods=["GET"])
def get_classes():
    """
    GET /api/classes

    Returns
    -------
    200 OK
    {
      "class_names": [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        ... (10 total)
      ]
    }

    Purpose: dashboard uses this to label confusion matrix columns.
    """
    cm_path = os.path.join(_ROOT, "models", "confusion_matrix.json")
    with open(cm_path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"class_names": data["class_names"]})


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Confusion matrix (for dashboard display)
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/api/confusion_matrix", methods=["GET"])
def get_confusion_matrix():
    """
    GET /api/confusion_matrix

    Returns
    -------
    200 OK  — full content of models/confusion_matrix.json:
    {
      "class_names": [...],
      "matrix":      [[int, ...], ...],   // 10x10
      "per_class":   {
        "ClassName": {
          "precision": float,
          "recall":    float,
          "f1":        float,
          "support":   int
        }, ...
      },
      "overall_accuracy": float,
      "macro_avg": {"precision": float, "recall": float, "f1": float}
    }

    Purpose: dashboard analytics panel — model evaluation stats.
    """
    cm_path = os.path.join(_ROOT, "models", "confusion_matrix.json")
    with open(cm_path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Sensor-only prediction
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/api/predict/sensor", methods=["POST"])
def predict_sensor():
    """
    POST /api/predict/sensor
    Content-Type: application/json

    Request body
    ------------
    {
      "soil_moisture_pct": float,   // 0-100
      "temperature_c":     float,   // -10 to 60
      "humidity_pct":      float    // 0-100
    }

    Response — 200 OK
    -----------------
    {
      "status":                   string,  // HOLD | IRRIGATE | SENSOR_ANOMALY
      "irrigate":                 bool,
      "sensor_irrigation_needed": int,     // 0 or 1 (XGBoost raw prediction)
      "sensor_probability":       float,   // P(irrigation_needed=1)
      "sensor_anomaly":           bool,
      "soil_moisture_pct":        float,
      "temperature_c":            float,
      "humidity_pct":             float,
      "disease_class":            null,
      "disease_confidence":       0.0,
      "disease_low_confidence":   false,
      "known_confusable_pair":    false,
      "top2_gap":                 null,
      "disease_stress_watch":     false,
      "treatment":                {},      // empty — no image
      "sensor_explanation": {
        "soil_moisture_pct": float,        // SHAP contribution
        "temperature_c":     float,
        "humidity_pct":      float
      },
      "image_explanation":  {},            // always empty for sensor-only
      "alerts":             [string, ...]  // may be empty list
    }

    Error — 400 Bad Request
    -----------------------
    { "error": "Bad Request", "detail": "Missing required field: 'soil_moisture_pct'" }
    """
    if not request.is_json:
        abort(400, "Content-Type must be application/json")

    data = request.get_json()
    try:
        moisture, temp, hum = _parse_sensor_fields(data)
    except ValueError as e:
        abort(400, str(e))

    shap_vals = _compute_shap(moisture, temp, hum)

    result: FusionResult = fuse(
        soil_moisture_pct  = moisture,
        temperature_c      = temp,
        humidity_pct       = hum,
        image_probs        = None,
        sensor_shap_values = shap_vals,
    )

    return jsonify(_to_native(asdict(result)))


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Full prediction (sensor + image)
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/api/predict/full", methods=["POST"])
def predict_full():
    """
    POST /api/predict/full
    Content-Type: multipart/form-data

    Form fields
    -----------
    soil_moisture_pct  (float, required)
    temperature_c      (float, required)
    humidity_pct       (float, required)
    image              (file,  required) — JPEG or PNG leaf photograph

    Response — 200 OK
    -----------------
    Same schema as /api/predict/sensor with these fields now populated:

      "status":                   string,  // all 7 status codes possible
      "disease_class":            string,  // e.g. "Tomato_Late_blight"
      "disease_confidence":       float,   // top-1 softmax probability
      "disease_low_confidence":   bool,    // true if < 0.70
      "known_confusable_pair":    bool,    // true if Target_Spot/Spider_mites gap < 0.35
      "top2_gap":                 float,   // P(top1) - P(top2)
      "disease_stress_watch":     bool,
      "treatment": {
        "display_name":         string,
        "pathogen":             string | null,
        "severity":             string,    // "none"|"moderate"|"high"
        "urgency":              string,    // "none"|"act_within_24h"|"act_within_48h"|"act_within_72h"
        "description":          string,
        "immediate_actions":    [string, ...],
        "chemical_treatments":  [
          {
            "product":   string,
            "dose":      string,
            "frequency": string,
            "icar_ref":  string   // may be absent
          }, ...
        ],
        "biological_treatments": [...],
        "irrigation_note":      string,
        "water_stress_link":    string,    // "none"|"monitor_closely"|"aggravated_by_drought"
        "icar_source":          string | null
      },
      "sensor_explanation": { ... },      // SHAP values (same as sensor-only)
      "image_explanation":  {}            // LIME not computed per-request (see note)

    NOTE on image_explanation
    -------------------------
    LIME explanations take ~60-120s per image. They are NOT computed here.
    The field is returned as {} so the frontend can render the prediction
    immediately. A separate endpoint (POST /api/explain/lime) can be added
    in a future iteration to request LIME asynchronously.

    Error — 400 Bad Request
    -----------------------
    { "error": "Bad Request", "detail": "..." }

    Error — 415 Unsupported Media Type (implicit via PIL)
    { "error": "Bad Request", "detail": "Image file is not a valid image." }
    """
    # ── Sensor fields (from form) ─────────────────────────────────────────────
    try:
        moisture, temp, hum = _parse_sensor_fields(request.form)
    except ValueError as e:
        abort(400, str(e))

    # ── Image file ────────────────────────────────────────────────────────────
    if "image" not in request.files:
        abort(400, "Missing required file field: 'image'")

    file = request.files["image"]
    if file.filename == "":
        abort(400, "Image file field is empty — no file was selected.")

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        pil_image.verify()          # catch corrupt files early
        pil_image = Image.open(io.BytesIO(img_bytes))  # re-open after verify
    except Exception:
        abort(400, "Image file is not a valid image (corrupt or unsupported format).")

    # ── Run image inference ───────────────────────────────────────────────────
    try:
        image_probs = img_inf.predict(pil_image)
    except Exception as e:
        abort(500, f"Image model inference failed: {e}")

    # ── SHAP for sensor ───────────────────────────────────────────────────────
    shap_vals = _compute_shap(moisture, temp, hum)

    # ── Fuse ──────────────────────────────────────────────────────────────────
    result: FusionResult = fuse(
        soil_moisture_pct  = moisture,
        temperature_c      = temp,
        humidity_pct       = hum,
        image_probs        = image_probs,
        sensor_shap_values = shap_vals,
    )

    return jsonify(asdict(result))


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Raw image probabilities (for debugging / frontend confidence bar)
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/api/classify/image", methods=["POST"])
def classify_image():
    """
    POST /api/classify/image
    Content-Type: multipart/form-data
    Form field: image (file)

    Returns the raw softmax probabilities for all 10 classes WITHOUT
    running fusion — useful for debugging the image model in isolation
    or for rendering a full probability bar chart on the dashboard.

    Response — 200 OK
    -----------------
    {
      "probabilities": {
        "Tomato_Bacterial_spot":                         float,
        "Tomato_Early_blight":                           float,
        "Tomato_Late_blight":                            float,
        "Tomato_Leaf_Mold":                              float,
        "Tomato_Septoria_leaf_spot":                     float,
        "Tomato_Spider_mites_Two_spotted_spider_mite":   float,
        "Tomato__Target_Spot":                           float,
        "Tomato__Tomato_YellowLeaf__Curl_Virus":         float,
        "Tomato__Tomato_mosaic_virus":                   float,
        "Tomato_healthy":                                float
      },
      "top_class":      string,
      "top_confidence": float
    }
    Probabilities are sorted descending by value.
    """
    if "image" not in request.files:
        abort(400, "Missing required file field: 'image'")

    file = request.files["image"]
    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        abort(400, "Image file is not a valid image.")

    probs = img_inf.predict(pil_image)
    top_class = next(iter(probs))
    return jsonify({
        "probabilities": probs,
        "top_class":     top_class,
        "top_confidence": probs[top_class],
    })


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Pre-load the sensor model at startup so the first request isn't slow.
    # Image model stays lazy — it's large and not always needed.
    print("Pre-loading sensor model...")
    try:
        _get_bundle()
        print("  Sensor model: OK")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    print("Starting Flask dev server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
