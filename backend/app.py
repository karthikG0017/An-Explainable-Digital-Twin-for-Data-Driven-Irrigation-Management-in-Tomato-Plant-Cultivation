"""
app.py
------
Item 6: Flask REST API for the Explainable Digital Twin irrigation system.

ENDPOINTS
=========

  GET  /health
  GET  /api/classes
  GET  /api/confusion_matrix
  GET  /api/history/<plant_id>
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

SHAP vs LIME vs Grad-CAM
========================
SHAP (sensor)  : computed per-request via TreeExplainer — ~5ms.
Grad-CAM (image): computed per-request via single backward pass — 53ms ±2ms.
                  Returns a base64 PNG heatmap overlay in `image_explanation`.
                  Benchmark (your CPU): LIME-1000=15.86s, LIME-300=4.44s (IoU=0.70±0.24),
                  LIME-100=1.49s (IoU=0.50±0.13), Grad-CAM=71ms total. Grad-CAM chosen.
LIME (offline) : 1000-sample LIME used in evaluation chapter (already computed).
                 Not run per-request — too slow and unstable at reduced sample counts.
"""

from __future__ import annotations

import os
import sys
import csv
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
import backend.gradcam_inference as gradcam_inf              # noqa: E402

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
      "image_explanation": {
        "method":          "gradcam",
        "predicted_class": string,         // top-1 class from Grad-CAM pass
        "confidence":      float,
        "heatmap_b64":     string,         // base64 JPEG (160x160) — embed as data:image/jpeg;base64,...
        "target_layer":    string,         // "features[-1] (EfficientNetB0 last MBConv block, 7x7)"
        "note":            string          // methodological note for UI display
      }

    Grad-CAM is computed per-request via a single backward pass (~53ms).
    Benchmark on this machine: LIME-1000=15.86s, LIME-300=4.44s (IoU=0.70+-0.24
    vs 1000-sample reference), Grad-CAM=71ms total. Grad-CAM chosen for live API.
    1000-sample LIME outputs are available as static assets from the evaluation chapter.

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

    # ── Run image inference (softmax probabilities) ───────────────────────────
    try:
        image_probs = img_inf.predict(pil_image)
    except Exception as e:
        abort(500, f"Image model inference failed: {e}")

    # ── Run Grad-CAM (~53ms) ──────────────────────────────────────────────────
    try:
        gradcam_result = gradcam_inf.explain(pil_image)
    except Exception as e:
        # Non-fatal: explanation failure should not block the prediction
        gradcam_result = {"method": "gradcam", "error": str(e)}

    # ── SHAP for sensor ───────────────────────────────────────────────────────
    shap_vals = _compute_shap(moisture, temp, hum)

    # ── Fuse ──────────────────────────────────────────────────────────────────
    result: FusionResult = fuse(
        soil_moisture_pct  = moisture,
        temperature_c      = temp,
        humidity_pct       = hum,
        image_probs        = image_probs,
        sensor_shap_values = shap_vals,
        image_lime_mask    = gradcam_result,   # stored in result.image_explanation
    )

    return jsonify(_to_native(asdict(result)))


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
# History endpoint (Item 7c)
# ════════════════════════════════════════════════════════════════════════════════

_history_cache = {}  # plant_id -> list of dicts

def _load_history(plant_id: str) -> list:
    """Load and cache simulated CSV data for a plant."""
    if plant_id in _history_cache:
        return _history_cache[plant_id]

    csv_path = os.path.join(_ROOT, "data", "processed", f"{plant_id}_labeled.csv")
    if not os.path.isfile(csv_path):
        return []

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp":        row["timestamp"],
                "soil_moisture_pct": float(row["soil_moisture_pct"]),
                "temperature_c":    float(row["temperature_c"]),
                "humidity_pct":     float(row["humidity_pct"]),
                "watered":          int(row.get("watered", 0)),
            })
    _history_cache[plant_id] = rows
    return rows


@app.route("/api/history/<plant_id>", methods=["GET"])
def get_history(plant_id):
    """
    GET /api/history/<plant_id>

    Returns historical sensor readings for a plant from simulated CSV data.

    Parameters
    ----------
    plant_id : str
        "plant_a" or "plant_b"

    Query parameters (optional)
    ---------------------------
    last : int
        Only return the last N data points (default: all).

    Response 200
    ------------
    {
      "plant_id": "plant_a",
      "count": 2016,
      "data": [
        {
          "timestamp": "2024-06-01T06:00:00",
          "soil_moisture_pct": 74.7,
          "temperature_c": 38.03,
          "humidity_pct": 41.79,
          "watered": 0
        },
        ...
      ]
    }
    """
    if plant_id not in ("plant_a", "plant_b"):
        abort(400, f"Invalid plant_id: {plant_id}. Use 'plant_a' or 'plant_b'.")

    rows = _load_history(plant_id)
    if not rows:
        abort(404, f"No history data found for {plant_id}.")

    # Optional: limit to last N data points
    last_n = request.args.get("last", type=int)
    if last_n and last_n > 0:
        rows = rows[-last_n:]

    return jsonify({
        "plant_id": plant_id,
        "count":    len(rows),
        "data":     rows,
    })


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
def _prewarm():
    """
    Pre-warm all models before Flask starts accepting requests so that the
    first real user request is not penalised by cold-load latency.

    Timing on this machine:
      Sensor model (XGBoost + IsoForest) : ~0.1s
      Image model (EfficientNetB0)        : ~3.5s  ← was hitting first request
      Grad-CAM dummy backward pass        : ~0.1s  ← also triggered here

    After pre-warming, all subsequent requests are ~70ms end-to-end.
    """
    import time

    # ── Sensor model ──────────────────────────────────────────────────────────
    print("Pre-warming sensor model (XGBoost + IsoForest)...")
    t0 = time.perf_counter()
    try:
        _get_bundle()
        print(f"  Sensor model: OK ({(time.perf_counter()-t0)*1000:.0f}ms)")
    except FileNotFoundError as e:
        print(f"  WARNING: sensor model not found — {e}")

    # ── Image model + Grad-CAM ────────────────────────────────────────────────
    # gradcam_inf and img_inf now share a single EfficientNetB0 singleton.
    # Calling img_inf._load() initialises the model; both modules then use it.
    # We then run one dummy forward pass (predict) and one dummy backward pass
    # (Grad-CAM) so both execution paths are JIT-compiled/cached by PyTorch.
    print("Pre-warming image model (EfficientNetB0) + Grad-CAM...")
    t0 = time.perf_counter()
    try:
        import numpy as np
        from PIL import Image as _Image

        dummy = _Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # 1. Load weights into memory (shared singleton)
        img_inf._load()

        # 2. Warm the forward path (img_inf.predict)
        img_inf.predict(dummy)

        # 3. Warm the backward path (Grad-CAM hooks + backward pass)
        gradcam_inf.explain(dummy)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Image model + Grad-CAM: OK ({elapsed_ms:.0f}ms)")
    except Exception as e:
        print(f"  WARNING: image model pre-warm failed — {e}")

    print()


if __name__ == "__main__":
    _prewarm()
    print("Starting server on http://localhost:5000")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000, threads=4)
    except ImportError:
        # Fallback to Flask dev server (not recommended for /predict/full
        # performance due to Werkzeug's synchronous multipart body parsing)
        print("(waitress not found — using Flask dev server; expect ~2s overhead on uploads)")
        app.run(host="0.0.0.0", port=5000, debug=False)
