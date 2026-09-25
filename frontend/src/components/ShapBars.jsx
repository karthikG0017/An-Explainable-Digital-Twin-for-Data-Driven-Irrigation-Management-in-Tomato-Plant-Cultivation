/**
 * ShapBars.jsx — Reusable SHAP value bar chart component.
 *
 * Extracted from ExplainabilityPanel (Item 7b) for reuse in WhatIfSimulator (7d).
 * Shows directional bars: green (positive = toward irrigation), blue (negative = away).
 */

const FEATURE_LABELS = {
  soil_moisture_pct: 'Soil Moisture',
  temperature_c: 'Temperature',
  humidity_pct: 'Humidity',
};

export default function ShapBars({ sensorExplanation }) {
  const hasSensorShap =
    sensorExplanation &&
    typeof sensorExplanation === 'object' &&
    !sensorExplanation.error &&
    Object.keys(sensorExplanation).length > 0;

  if (!hasSensorShap) {
    return (
      <div className="explain-empty">
        {sensorExplanation?.error
          ? `SHAP error: ${sensorExplanation.error}`
          : 'No sensor explanation available'}
      </div>
    );
  }

  let maxAbsShap = 0;
  for (const key of Object.keys(FEATURE_LABELS)) {
    const val = Math.abs(sensorExplanation[key] || 0);
    if (val > maxAbsShap) maxAbsShap = val;
  }

  return (
    <div className="shap-bars">
      {Object.entries(FEATURE_LABELS).map(([key, label]) => {
        const value = sensorExplanation[key] || 0;
        const isPositive = value >= 0;
        const barWidth = maxAbsShap > 0
          ? (Math.abs(value) / maxAbsShap) * 100
          : 0;

        return (
          <div key={key} className="shap-row">
            <div className="shap-label">{label}</div>
            <div className="shap-bar-container">
              <div className="shap-bar-negative">
                {!isPositive && (
                  <div
                    className="shap-bar shap-bar-neg"
                    style={{ width: `${barWidth}%` }}
                  />
                )}
              </div>
              <div className="shap-bar-axis" />
              <div className="shap-bar-positive">
                {isPositive && (
                  <div
                    className="shap-bar shap-bar-pos"
                    style={{ width: `${barWidth}%` }}
                  />
                )}
              </div>
            </div>
            <div className={`shap-value ${isPositive ? 'shap-pos' : 'shap-neg'}`}>
              {isPositive ? '↑' : '↓'} {value.toFixed(2)}
            </div>
          </div>
        );
      })}
      <div className="shap-legend">
        <span className="shap-legend-neg">← Less irrigation</span>
        <span className="shap-legend-pos">More irrigation →</span>
      </div>
    </div>
  );
}
