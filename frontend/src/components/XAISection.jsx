import React from 'react';
import { Info, TrendingUp } from 'lucide-react';
import './XAISection.css';

function XAISection({ images, anomaly, prediction }) {
  const getAnomalyLevel = (score) => {
    if (score < 30) return { level: 'Low', color: '#10b981' };
    if (score < 70) return { level: 'Medium', color: '#f59e0b' };
    return { level: 'High', color: '#ef4444' };
  };

  const anomalyInfo = getAnomalyLevel(anomaly.score);

  return (
    <div className="xai-section card">
      <div className="section-header">
        <h2>Explainable AI - Feature Analysis</h2>
        <div className="info-badge">
          <Info size={16} />
          <span>Activation-based Anomaly Detection</span>
        </div>
      </div>

      <div className="anomaly-metrics">
        <div className="metric-card">
          <div className="metric-label">Anomaly Score</div>
          <div className="metric-value" style={{ color: anomalyInfo.color }}>
            {anomaly.score.toFixed(2)}
          </div>
          <div className="metric-badge" style={{ background: `${anomalyInfo.color}20`, color: anomalyInfo.color }}>
            {anomalyInfo.level} Severity
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Mean Anomaly</div>
          <div className="metric-value">
            {anomaly.mean_score.toFixed(2)}
          </div>
          <div className="metric-description">
            Average across all regions
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Defect Type</div>
          <div className="metric-value defect-type">
            {prediction.class}
          </div>
          <div className="metric-description">
            Detected classification
          </div>
        </div>
      </div>

      <div className="images-comparison">
        <div className="comparison-item">
          <div className="comparison-label">Original Image</div>
          <div className="image-wrapper-small">
            <img src={images.original} alt="Original" className="viz-image-small" />
          </div>
        </div>

        <div className="comparison-item">
          <div className="comparison-label">Anomaly Heatmap</div>
          <div className="image-wrapper-small">
            <img src={images.heatmap} alt="Heatmap" className="viz-image-small" />
          </div>
        </div>
      </div>

      <div className="legend">
        <div className="legend-title">Heatmap Legend</div>
        <div className="legend-gradient">
          <span>Normal</span>
          <div className="gradient-bar"></div>
          <span>Defect</span>
        </div>
        <div className="legend-description">
          Warmer colors (red/yellow) indicate regions with high anomaly scores where defects are detected
        </div>
      </div>

      <div className="interpretation-panel">
        <div className="interpretation-header">
          <TrendingUp size={20} />
          <h3>Interpretation</h3>
        </div>
        <div className="interpretation-content">
          <p>
            The heatmap visualizes where the neural network detects anomalous features by analyzing 
            activation patterns across multiple layers.
          </p>
          <ul>
            <li><strong>Red/Yellow regions:</strong> High anomaly scores - defect locations detected</li>
            <li><strong>Blue/Purple regions:</strong> Low anomaly scores - normal surface areas</li>
            <li><strong>Score interpretation:</strong> Higher values indicate stronger deviation from normal patterns</li>
          </ul>
          <p className="note">
            💡 This visualization helps quality inspectors understand <em>where</em> and 
            <em>why</em> the AI detected a defect, improving trust and decision-making.
          </p>
        </div>
      </div>
    </div>
  );
}

export default XAISection;
