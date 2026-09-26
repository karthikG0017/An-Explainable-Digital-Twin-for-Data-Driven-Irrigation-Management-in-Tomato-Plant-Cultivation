/**
 * ABComparison.jsx — Side-by-side A/B comparison of Plant A vs Plant B.
 *
 * Item 7e: Shows summary metrics from historical (simulated) data,
 * highlighting irrigation efficiency and plant health outcomes.
 *
 * Visual treatment: solid teal/green border (NOT amber/dashed like What-If).
 * This presents real simulated results, not hypothetical scenarios.
 */
import { useState, useEffect } from 'react';

const API_BASE = '';

export default function ABComparison() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/comparison`);
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const json = await res.json();
        setData(json);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="ab-section">
        <div className="ab-loading">Loading comparison data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ab-section">
        <div className="ab-error">⚠️ {error}</div>
      </div>
    );
  }

  if (!data) return null;

  const { plant_a: a, plant_b: b, note } = data;

  // Determine "winner" for each metric
  const betterMoisture = a.avg_moisture > b.avg_moisture ? 'a' : 'b';
  const fewerEvents = a.irrigation_events < b.irrigation_events ? 'a' : 'b';
  const moreHealthy = a.healthy_pct > b.healthy_pct ? 'a' : 'b';

  return (
    <div className="ab-section">
      <div className="ab-header">
        <div className="ab-title">
          <span className="ab-icon">📊</span>
          A/B Comparison — Irrigation Strategy Outcomes
          <span className="ab-badge">SIMULATED RESULTS</span>
        </div>
        <div className="ab-subtitle">
          Plant A (fixed-schedule) vs Plant B (AI-managed) over 6 weeks of simulated data
        </div>
      </div>

      <div className="ab-body">
        {/* ── Summary insight ─────────────────────────────────── */}
        <div className="ab-insight">
          Plant B (AI-managed) achieved comparable plant health
          ({b.healthy_pct}% vs {a.healthy_pct}% healthy time) with
          fewer irrigation events ({b.irrigation_events} vs {a.irrigation_events}) than
          Plant A's fixed schedule — a modest efficiency gain using
          threshold-based decision logic.
        </div>

        {/* ── Comparison table ────────────────────────────────── */}
        <table className="ab-table">
          <thead>
            <tr>
              <th className="ab-metric-col">Metric</th>
              <th className="ab-plant-col ab-col-a">
                <span className="ab-plant-dot ab-dot-a" />
                Plant A
                <span className="ab-plant-type">Fixed Schedule</span>
              </th>
              <th className="ab-plant-col ab-col-b">
                <span className="ab-plant-dot ab-dot-b" />
                Plant B
                <span className="ab-plant-type">AI-Managed</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="ab-metric-name">Avg Soil Moisture</td>
              <td className={`ab-cell ${betterMoisture === 'a' ? 'ab-winner' : ''}`}>
                {a.avg_moisture}%
              </td>
              <td className={`ab-cell ${betterMoisture === 'b' ? 'ab-winner' : ''}`}>
                {b.avg_moisture}%
              </td>
            </tr>
            <tr>
              <td className="ab-metric-name">Min Moisture</td>
              <td className="ab-cell">{a.min_moisture}%</td>
              <td className="ab-cell">{b.min_moisture}%</td>
            </tr>
            <tr>
              <td className="ab-metric-name">Irrigation Events</td>
              <td className={`ab-cell ${fewerEvents === 'a' ? 'ab-winner' : ''}`}>
                {a.irrigation_events}
              </td>
              <td className={`ab-cell ${fewerEvents === 'b' ? 'ab-winner' : ''}`}>
                {b.irrigation_events}
                {fewerEvents === 'b' && (
                  <span className="ab-delta">
                    {' '}({Math.round((1 - b.irrigation_events / a.irrigation_events) * 100)}% fewer)
                  </span>
                )}
              </td>
            </tr>
            <tr className="ab-tier-row">
              <td className="ab-metric-name">
                <span className="ab-tier-icon">🟢</span> Healthy Time
              </td>
              <td className={`ab-cell ${moreHealthy === 'a' ? 'ab-winner' : ''}`}>
                {a.healthy_pct}%
              </td>
              <td className={`ab-cell ${moreHealthy === 'b' ? 'ab-winner' : ''}`}>
                {b.healthy_pct}%
              </td>
            </tr>
            <tr className="ab-tier-row">
              <td className="ab-metric-name">
                <span className="ab-tier-icon">🟡</span> Moderate Stress
              </td>
              <td className="ab-cell">{a.moderate_stress_pct}%</td>
              <td className="ab-cell">{b.moderate_stress_pct}%</td>
            </tr>
            <tr className="ab-tier-row">
              <td className="ab-metric-name">
                <span className="ab-tier-icon">🔴</span> High Stress
              </td>
              <td className="ab-cell">{a.high_stress_pct}%</td>
              <td className="ab-cell">{b.high_stress_pct}%</td>
            </tr>
          </tbody>
        </table>

        {/* ── Stress tier visual bars ─────────────────────────── */}
        <div className="ab-bars-section">
          <div className="ab-bars-title">Stress Tier Distribution</div>
          <div className="ab-bars-row">
            <span className="ab-bars-label">Plant A</span>
            <div className="ab-stacked-bar">
              <div
                className="ab-bar-seg ab-seg-healthy"
                style={{ width: `${a.healthy_pct}%` }}
                title={`Healthy: ${a.healthy_pct}%`}
              />
              <div
                className="ab-bar-seg ab-seg-moderate"
                style={{ width: `${a.moderate_stress_pct}%` }}
                title={`Moderate: ${a.moderate_stress_pct}%`}
              />
              <div
                className="ab-bar-seg ab-seg-high"
                style={{ width: `${a.high_stress_pct}%` }}
                title={`High: ${a.high_stress_pct}%`}
              />
            </div>
          </div>
          <div className="ab-bars-row">
            <span className="ab-bars-label">Plant B</span>
            <div className="ab-stacked-bar">
              <div
                className="ab-bar-seg ab-seg-healthy"
                style={{ width: `${b.healthy_pct}%` }}
                title={`Healthy: ${b.healthy_pct}%`}
              />
              <div
                className="ab-bar-seg ab-seg-moderate"
                style={{ width: `${b.moderate_stress_pct}%` }}
                title={`Moderate: ${b.moderate_stress_pct}%`}
              />
              <div
                className="ab-bar-seg ab-seg-high"
                style={{ width: `${b.high_stress_pct}%` }}
                title={`High: ${b.high_stress_pct}%`}
              />
            </div>
          </div>
          <div className="ab-bars-legend">
            <span><span className="ab-legend-swatch ab-seg-healthy" /> Healthy</span>
            <span><span className="ab-legend-swatch ab-seg-moderate" /> Moderate</span>
            <span><span className="ab-legend-swatch ab-seg-high" /> High</span>
          </div>
        </div>
      </div>

      {/* ── Disclaimer note ───────────────────────────────────── */}
      <div className="ab-note">
        <span className="ab-note-icon">ℹ️</span>
        {note}
      </div>
    </div>
  );
}
