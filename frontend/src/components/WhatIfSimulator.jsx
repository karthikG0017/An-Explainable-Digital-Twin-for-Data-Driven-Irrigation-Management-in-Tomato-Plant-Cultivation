/**
 * WhatIfSimulator.jsx — Hypothetical sensor scenario simulator.
 *
 * Item 7d: Standalone section below the plant cards.
 * Calls POST /api/predict/sensor with user-set values and shows
 * the resulting status, irrigate decision, and SHAP bars (reused from ShapBars).
 *
 * Visually distinct from live cards — uses dashed border, "SIMULATION" badge,
 * and a different background to prevent confusion with real readings.
 */
import { useState, useCallback } from 'react';
import ShapBars from './ShapBars';

const API_BASE = '';

/**
 * Compute moisture-based urgency tier from raw moisture value.
 * These tiers were defined in item 2 (label engineering) and provide
 * gradation that the XGBoost step-function decision cannot.
 */
function getUrgencyTier(moisturePct) {
  if (moisturePct >= 70) return { label: 'Healthy',         level: 'healthy',         icon: '🟢', desc: 'Soil moisture adequate (≥70%)' };
  if (moisturePct >= 50) return { label: 'Moderate Stress',  level: 'moderate_stress', icon: '🟡', desc: `Moisture at ${moisturePct}% — below 70% threshold` };
  return                         { label: 'High Stress',     level: 'high_stress',     icon: '🔴', desc: `Moisture critically low at ${moisturePct}% — well below 50%` };
}

export default function WhatIfSimulator() {
  const [moisture, setMoisture] = useState(65);
  const [temperature, setTemperature] = useState(30);
  const [humidity, setHumidity] = useState(60);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const simulate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/predict/sensor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          soil_moisture_pct: moisture,
          temperature_c: temperature,
          humidity_pct: humidity,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `API error: ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [moisture, temperature, humidity]);

  return (
    <div className="whatif-section">
      <div className="whatif-header">
        <div className="whatif-title">
          <span className="whatif-icon">🧪</span>
          What-If Simulator
          <span className="whatif-badge">HYPOTHETICAL</span>
        </div>
        <div className="whatif-subtitle">
          Adjust sensor values below and simulate the AI irrigation decision.
          This does NOT affect real plants — it's a sandbox for exploring model behavior.
        </div>
      </div>

      <div className="whatif-body">
        {/* ── Input controls ────────────────────────────────────── */}
        <div className="whatif-controls">
          <div className="whatif-slider-group">
            <label className="whatif-label" htmlFor="sim-moisture">
              Soil Moisture
              <span className="whatif-val">{moisture.toFixed(0)}%</span>
            </label>
            <input
              id="sim-moisture"
              type="range"
              min="0"
              max="100"
              step="1"
              value={moisture}
              onChange={(e) => setMoisture(Number(e.target.value))}
              className="whatif-slider whatif-slider-moisture"
            />
            <div className="whatif-range-labels">
              <span>0%</span>
              <span className="whatif-threshold-mark">70% threshold</span>
              <span>100%</span>
            </div>
          </div>

          <div className="whatif-slider-group">
            <label className="whatif-label" htmlFor="sim-temp">
              Temperature
              <span className="whatif-val">{temperature.toFixed(0)}°C</span>
            </label>
            <input
              id="sim-temp"
              type="range"
              min="10"
              max="50"
              step="1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="whatif-slider whatif-slider-temp"
            />
            <div className="whatif-range-labels">
              <span>10°C</span>
              <span>50°C</span>
            </div>
          </div>

          <div className="whatif-slider-group">
            <label className="whatif-label" htmlFor="sim-humidity">
              Humidity
              <span className="whatif-val">{humidity.toFixed(0)}%</span>
            </label>
            <input
              id="sim-humidity"
              type="range"
              min="0"
              max="100"
              step="1"
              value={humidity}
              onChange={(e) => setHumidity(Number(e.target.value))}
              className="whatif-slider whatif-slider-humidity"
            />
            <div className="whatif-range-labels">
              <span>0%</span>
              <span>100%</span>
            </div>
          </div>

          <button
            className="whatif-simulate-btn"
            onClick={simulate}
            disabled={loading}
          >
            {loading ? '⟳ Simulating...' : '▶ Simulate'}
          </button>
        </div>

        {/* ── Results ───────────────────────────────────────────── */}
        <div className="whatif-results">
          {error && (
            <div className="whatif-error">⚠️ {error}</div>
          )}

          {!result && !error && !loading && (
            <div className="whatif-placeholder">
              Adjust the sliders and click <strong>Simulate</strong> to see the AI's decision and SHAP explanation.
            </div>
          )}

          {loading && (
            <div className="whatif-placeholder">Running simulation...</div>
          )}

          {result && !loading && (
            <>
              <div className="whatif-result-header">
                <div className={`whatif-status-badge whatif-status-${result.status}`}>
                  {result.status.replace(/_/g, ' ')}
                </div>
                <div className="whatif-decision">
                  {result.irrigate
                    ? '💧 Model recommends: IRRIGATE'
                    : '✋ Model recommends: HOLD'}
                </div>
                <div className="whatif-probability">
                  Irrigation probability: {(result.sensor_probability * 100).toFixed(1)}%
                </div>
              </div>

              {/* Urgency tier — compensates for model's step-function limitation */}
              {(() => {
                const tier = getUrgencyTier(moisture);
                return (
                  <div className={`whatif-urgency whatif-urgency-${tier.level}`}>
                    <span className="whatif-urgency-icon">{tier.icon}</span>
                    <div className="whatif-urgency-content">
                      <span className="whatif-urgency-label">Moisture Stress Tier: {tier.label}</span>
                      <span className="whatif-urgency-desc">{tier.desc}</span>
                    </div>
                  </div>
                );
              })()}

              <div className="whatif-shap-section">
                <div className="whatif-shap-title">
                  SHAP Explanation
                  <span className="explain-subtitle">Why the model made this decision</span>
                </div>
                <ShapBars sensorExplanation={result.sensor_explanation} />
              </div>

              {result.sensor_anomaly && (
                <div className="whatif-anomaly-note">
                  🚨 These readings were flagged as anomalous by the Isolation Forest.
                  In production, the irrigation decision would be suppressed.
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
