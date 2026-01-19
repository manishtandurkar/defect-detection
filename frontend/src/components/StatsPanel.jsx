import React from 'react';
import { Clock, TrendingUp, Activity } from 'lucide-react';
import './StatsPanel.css';

function StatsPanel({ history }) {
  const getDefectStats = () => {
    const stats = {};
    history.forEach(item => {
      stats[item.prediction] = (stats[item.prediction] || 0) + 1;
    });
    return stats;
  };

  const defectStats = getDefectStats();
  const totalAnalyzed = history.length;

  return (
    <div className="stats-panel">
      <div className="card stats-card">
        <h3>
          <Activity size={20} />
          Analysis Statistics
        </h3>
        
        <div className="stat-item">
          <div className="stat-label">Total Analyzed</div>
          <div className="stat-value">{totalAnalyzed}</div>
        </div>

        {totalAnalyzed > 0 && (
          <div className="defect-breakdown">
            <div className="breakdown-title">Defect Breakdown</div>
            {Object.entries(defectStats).map(([defect, count]) => (
              <div key={defect} className="breakdown-item">
                <span className="breakdown-label">{defect}</span>
                <div className="breakdown-bar-container">
                  <div 
                    className="breakdown-bar"
                    style={{ width: `${(count / totalAnalyzed) * 100}%` }}
                  />
                </div>
                <span className="breakdown-count">{count}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card history-card">
        <h3>
          <Clock size={20} />
          Recent History
        </h3>
        
        {history.length === 0 ? (
          <div className="empty-state">
            <TrendingUp size={32} opacity={0.3} />
            <p>No analyses yet</p>
          </div>
        ) : (
          <div className="history-list">
            {history.map(item => (
              <div key={item.id} className="history-item">
                <div className="history-header">
                  <span className="history-defect">{item.prediction}</span>
                  <span className="history-confidence">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="history-footer">
                  <span className="history-time">{item.timestamp}</span>
                  <span className="history-anomaly">
                    Anomaly: {item.anomalyScore.toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default StatsPanel;
