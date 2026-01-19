import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { CheckCircle, AlertTriangle } from 'lucide-react';
import './ResultsSection.css';

function ResultsSection({ predictions, originalImage }) {
  const { prediction } = predictions;
  
  // Prepare data for chart
  const chartData = Object.entries(prediction.all_probabilities).map(([name, value]) => ({
    name,
    probability: (value * 100).toFixed(2),
    value: value
  }));

  // Sort by probability
  chartData.sort((a, b) => b.value - a.value);

  const getBarColor = (name) => {
    return name === prediction.class ? '#667eea' : '#cbd5e0';
  };

  return (
    <div className="results-section card">
      <h2>Detection Results</h2>
      
      <div className="results-grid">
        <div className="result-card primary">
          <div className="result-icon">
            {prediction.confidence > 0.8 ? (
              <CheckCircle size={32} color="#10b981" />
            ) : (
              <AlertTriangle size={32} color="#f59e0b" />
            )}
          </div>
          <div className="result-info">
            <div className="result-label">Detected Defect</div>
            <div className="result-value">{prediction.class}</div>
            <div className="confidence-bar">
              <div 
                className="confidence-fill" 
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
            <div className="confidence-text">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      <div className="probability-chart">
        <h3>Class Probabilities</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis dataKey="name" type="category" width={100} />
            <Tooltip formatter={(value) => `${value}%`} />
            <Bar dataKey="probability" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.name)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default ResultsSection;
