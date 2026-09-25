/**
 * ExplainabilityPanel.jsx — SHAP sensor explanation + Grad-CAM image explanation.
 *
 * Item 7b: Renders per-plant within each PlantCard.
 *
 * sensor_explanation: { soil_moisture_pct: float, temperature_c: float, humidity_pct: float }
 *   SHAP values — positive pushes toward irrigation, negative pushes away.
 *
 * image_explanation: {
 *   method: "gradcam",
 *   predicted_class: string,
 *   confidence: float,
 *   heatmap_b64: string (base64 JPEG),
 *   target_layer: string,
 *   note: string
 * }
 *   Empty {} when no image was analyzed.
 */
import ShapBars from './ShapBars';

/**
 * Format disease class name for display in Grad-CAM caption.
 */
function formatClassName(className) {
  if (!className) return '—';
  if (className === 'Tomato_healthy') return 'Healthy';
  return className
    .replace(/^Tomato_+/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function ExplainabilityPanel({ sensorExplanation, imageExplanation }) {
  const hasGradCam =
    imageExplanation &&
    typeof imageExplanation === 'object' &&
    imageExplanation.method === 'gradcam' &&
    imageExplanation.heatmap_b64;

  const hasImageExplanation =
    imageExplanation &&
    typeof imageExplanation === 'object' &&
    Object.keys(imageExplanation).length > 0;

  return (
    <div className="explainability-panel">
      <div className="explain-header">Explainability</div>

      <div className="explain-grid">
        {/* ── SHAP: Sensor explanation ─────────────────────────────────── */}
        <div className="explain-section">
          <div className="explain-section-title">
            Sensor SHAP Values
            <span className="explain-subtitle">Feature contribution to irrigation decision</span>
          </div>
          <ShapBars sensorExplanation={sensorExplanation} />
        </div>

        {/* ── Grad-CAM: Image explanation ──────────────────────────────── */}
        <div className="explain-section">
          <div className="explain-section-title">
            Image Explanation
            <span className="explain-subtitle">Grad-CAM heatmap overlay</span>
          </div>

          {hasGradCam ? (
            <div className="gradcam-content">
              <div className="gradcam-image-wrapper">
                <img
                  src={`data:image/jpeg;base64,${imageExplanation.heatmap_b64}`}
                  alt={`Grad-CAM heatmap for ${imageExplanation.predicted_class}`}
                  className="gradcam-image"
                />
              </div>
              <div className="gradcam-meta">
                <div className="gradcam-prediction">
                  <span className="gradcam-meta-label">Predicted:</span>
                  {formatClassName(imageExplanation.predicted_class)}
                  <span className="gradcam-conf">
                    ({(imageExplanation.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="gradcam-layer">
                  <span className="gradcam-meta-label">Layer:</span>
                  {imageExplanation.target_layer}
                </div>
                {imageExplanation.note && (
                  <div className="gradcam-note">{imageExplanation.note}</div>
                )}
              </div>
            </div>
          ) : hasImageExplanation ? (
            /* image_explanation exists but has no heatmap (e.g., error or partial) */
            <div className="explain-empty">
              Image explanation incomplete — no heatmap available
            </div>
          ) : (
            /* image_explanation is empty {} — no image was provided */
            <div className="explain-empty explain-no-image">
              <div className="explain-no-image-icon">📷</div>
              No image analyzed for this reading
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
