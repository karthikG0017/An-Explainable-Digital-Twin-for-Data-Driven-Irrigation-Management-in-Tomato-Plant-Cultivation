/**
 * PlantCard.jsx — Displays live status for a single plant.
 *
 * Shows: status badge, sensor readings, disease info, all alerts,
 * SHAP + Grad-CAM explainability panel (Item 7b),
 * and historical trend chart (Item 7c).
 */
import ExplainabilityPanel from './ExplainabilityPanel';
import HistoryChart from './HistoryChart';

/**
 * Classify alert type for styling based on the alert tag prefix.
 */
function getAlertStyle(alertText) {
  if (alertText.startsWith('SENSOR_ANOMALY')) return 'alert-danger';
  if (alertText.startsWith('DOMAIN_GAP_CAUTION')) return 'alert-danger';
  if (alertText.startsWith('IMAGE_LOW_CONFIDENCE')) return 'alert-warning';
  if (alertText.startsWith('UNCERTAIN_PAIR')) return 'alert-warning';
  if (alertText.startsWith('DISEASE_STRESS_WATCH')) return 'alert-warning';
  return 'alert-info';
}

/**
 * Extract the tag prefix (e.g. "SENSOR_ANOMALY") from an alert string.
 */
function getAlertTag(alertText) {
  const colonIdx = alertText.indexOf(':');
  if (colonIdx > 0 && colonIdx < 40) {
    return alertText.substring(0, colonIdx);
  }
  return 'ALERT';
}

/**
 * Format disease class name for display.
 * "Tomato_Early_blight" → "Early Blight"
 * "Tomato__Tomato_YellowLeaf__Curl_Virus" → "Yellow Leaf Curl Virus"
 */
function formatDiseaseName(className) {
  if (!className) return 'None detected';
  if (className === 'Tomato_healthy') return 'Healthy';
  return className
    .replace(/^Tomato_+/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PlantCard({ plantId, label, typeLabel, data, error, loading }) {
  // ── Loading state ──────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="plant-card">
        <div className="plant-card-header">
          <div className="plant-label">
            {label}
            <span className="plant-type">{typeLabel}</span>
          </div>
        </div>
        <div className="plant-card-body">
          <div className="state-message state-loading">
            <div className="icon">⏳</div>
            <div className="title">Loading...</div>
            <div className="detail">Fetching latest readings</div>
          </div>
        </div>
      </div>
    );
  }

  // ── Error state ────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="plant-card">
        <div className="plant-card-header">
          <div className="plant-label">
            {label}
            <span className="plant-type">{typeLabel}</span>
          </div>
        </div>
        <div className="plant-card-body">
          <div className="state-message state-error">
            <div className="icon">⚠️</div>
            <div className="title">Backend Unreachable</div>
            <div className="detail">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const isHealthy = data.disease_class === 'Tomato_healthy' || !data.disease_class;
  const hasDisease = data.disease_class && data.disease_class !== 'Tomato_healthy';

  return (
    <div className="plant-card">
      {/* Header: plant label + status badge */}
      <div className="plant-card-header">
        <div className="plant-label">
          {label}
          <span className="plant-type">{typeLabel}</span>
        </div>
        <div className={`status-badge status-${data.status}`}>
          {data.status.replace(/_/g, ' ')}
        </div>
      </div>

      <div className="plant-card-body">
        {/* Anomaly banner — prominent, top of card body */}
        {data.sensor_anomaly && (
          <div className="anomaly-banner">
            🚨 Sensor reading flagged as anomalous — check hardware
          </div>
        )}

        {/* Sensor readings */}
        <div className="sensor-readings">
          <div className="sensor-reading moisture">
            <div className="label">Soil Moisture</div>
            <div className="value">
              {data.soil_moisture_pct.toFixed(1)}
              <span className="unit">%</span>
            </div>
          </div>
          <div className="sensor-reading temperature">
            <div className="label">Temperature</div>
            <div className="value">
              {data.temperature_c.toFixed(1)}
              <span className="unit">°C</span>
            </div>
          </div>
          <div className="sensor-reading humidity">
            <div className="label">Humidity</div>
            <div className="value">
              {data.humidity_pct.toFixed(1)}
              <span className="unit">%</span>
            </div>
          </div>
        </div>

        {/* Disease info — three distinct states:
             1. disease_class=null     → no image was analyzed
             2. disease_class=healthy  → image analyzed, plant looks healthy
             3. disease_class=disease  → disease detected with confidence */}
        <div className={`disease-section ${hasDisease ? 'has-disease' : 'healthy'}`}>
          <div className="disease-label">Disease Detection</div>
          <div className="disease-name">{formatDiseaseName(data.disease_class)}</div>
          {hasDisease && (
            <div className="disease-confidence">
              Confidence: {(data.disease_confidence * 100).toFixed(1)}%
              {data.disease_low_confidence && ' (low)'}
              {data.known_confusable_pair && ' · confusable pair flagged'}
            </div>
          )}
          {data.disease_class === 'Tomato_healthy' && (
            <div className="disease-confidence">
              Image analyzed — no disease detected (confidence: {(data.disease_confidence * 100).toFixed(1)}%)
            </div>
          )}
          {!data.disease_class && (
            <div className="disease-confidence">No image provided for this reading</div>
          )}
        </div>

        {/* Alerts — render every single one, never drop or summarize */}
        {data.alerts && data.alerts.length > 0 && (
          <div className="alerts-section">
            <div className="alerts-title">
              Alerts ({data.alerts.length})
            </div>
            {data.alerts.map((alert, idx) => (
              <div
                key={idx}
                className={`alert-item ${getAlertStyle(alert)}`}
              >
                <span className="alert-tag">{getAlertTag(alert)}:</span>
                {alert.substring(alert.indexOf(':') + 2)}
              </div>
            ))}
          </div>
        )}

        {/* Item 7b: SHAP sensor + Grad-CAM image explainability */}
        <ExplainabilityPanel
          sensorExplanation={data.sensor_explanation}
          imageExplanation={data.image_explanation}
        />

        {/* Item 7c: Historical trend chart */}
        <HistoryChart plantId={plantId} />
      </div>
    </div>
  );
}
