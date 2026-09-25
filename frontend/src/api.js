/**
 * api.js — API helpers for the Digital Twin dashboard.
 *
 * Calls POST /api/predict/full with sensor readings + leaf image.
 * When USE_MOCK=true, returns mock data instead of calling the backend
 * (for frontend development before ESP32 data is flowing).
 */

const API_BASE = ''; // proxied by Vite in dev

// ── Toggle this to switch between mock and live data ────────────────────
const USE_MOCK = true;

// ── Mock data matching the exact FusionResult schema ────────────────────
// Plant A: sensor-only (no image) — tests SHAP bars with no Grad-CAM
const MOCK_PLANT_A = {
  status: 'HOLD',
  irrigate: false,
  sensor_irrigation_needed: 0,
  sensor_probability: 0.0003,
  sensor_anomaly: false,
  soil_moisture_pct: 78.4,
  temperature_c: 29.2,
  humidity_pct: 68.5,
  disease_class: null,
  disease_confidence: 0.0,
  disease_low_confidence: false,
  known_confusable_pair: false,
  top2_gap: null,
  disease_stress_watch: false,
  treatment: {},
  sensor_explanation: {
    soil_moisture_pct: -7.50,
    temperature_c: -0.28,
    humidity_pct: -0.30,
  },
  image_explanation: {},
  alerts: [],
};

// Plant B: full prediction with Grad-CAM — tests SHAP + heatmap rendering
// heatmap_b64 will be empty string for mock (renders as black box),
// but real backend returns actual base64 JPEG content.
const MOCK_PLANT_B = {
  status: 'IRRIGATE_WITH_ALERT',
  irrigate: true,
  sensor_irrigation_needed: 1,
  sensor_probability: 0.8741,
  sensor_anomaly: false,
  soil_moisture_pct: 42.1,
  temperature_c: 33.6,
  humidity_pct: 55.2,
  disease_class: 'Tomato_Early_blight',
  disease_confidence: 0.538,
  disease_low_confidence: true,
  known_confusable_pair: false,
  top2_gap: 0.214,
  disease_stress_watch: false,
  treatment: {
    display_name: 'Early Blight',
    pathogen: 'Alternaria solani',
    severity: 'moderate',
    urgency: 'act_within_48h',
  },
  sensor_explanation: {
    soil_moisture_pct: 7.60,
    temperature_c: 0.52,
    humidity_pct: -0.11,
  },
  image_explanation: {
    method: 'gradcam',
    predicted_class: 'Tomato_Early_blight',
    confidence: 0.538,
    heatmap_b64: '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',  // real Grad-CAM from backend
    target_layer: 'features[-1] (EfficientNetB0 last MBConv block, 7x7)',
    note: 'Grad-CAM highlights regions the model attended to for this prediction.',
  },
  alerts: [
    "IMAGE_LOW_CONFIDENCE: Disease prediction 'Tomato_Early_blight' has confidence 53.8% (below 70% threshold). Treat as uncertain — do not report to grower as definitive.",
    "DOMAIN_GAP_CAUTION: Disease 'Tomato_Early_blight' flagged at 53.8% confidence on a plant with healthy sensor readings (moisture=42.1%, no irrigation needed). Our model's real-world healthy recall is ~27%, meaning it may misclassify healthy field images as diseased. Recommend visual confirmation before treatment.",
  ],
};

/**
 * Fetch prediction for a plant.
 * @param {'A'|'B'} plantId - Which plant to query.
 * @returns {Promise<Object>} FusionResult-shaped object.
 */
export async function fetchPlantStatus(plantId) {
  if (USE_MOCK) {
    // Simulate network delay
    await new Promise((r) => setTimeout(r, 300 + Math.random() * 200));
    return plantId === 'A' ? { ...MOCK_PLANT_A } : { ...MOCK_PLANT_B };
  }

  // Live mode: call POST /api/predict/sensor with sample readings per plant.
  // Plant A (control): well-watered, should get HOLD
  // Plant B (AI-managed): stressed soil, should get IRRIGATE
  // TODO: Replace with live Firebase readings when ESP32 is active.
  // When image upload is wired (7b), switch to /api/predict/full with FormData.
  const sensorData =
    plantId === 'A'
      ? { soil_moisture_pct: 78.4, temperature_c: 29.2, humidity_pct: 68.5 }
      : { soil_moisture_pct: 42.1, temperature_c: 33.6, humidity_pct: 55.2 };

  const res = await fetch(`${API_BASE}/api/predict/sensor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sensorData),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }

  return res.json();
}

/**
 * Check backend health.
 * @returns {Promise<Object>} Health status.
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * Fetch historical sensor readings for a plant.
 * @param {'A'|'B'} plantId - Which plant to query.
 * @param {number} [last] - Optional: only fetch the last N data points.
 * @returns {Promise<Object>} { plant_id, count, data: [...] }
 */
export async function fetchHistory(plantId, last) {
  const apiPlantId = plantId === 'A' ? 'plant_a' : 'plant_b';
  let url = `${API_BASE}/api/history/${apiPlantId}`;
  if (last) url += `?last=${last}`;

  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `History API error: ${res.status}`);
  }
  return res.json();
}
