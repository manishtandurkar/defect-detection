import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import './ResultsSection.css';

function ResultsSection({ predictions, originalImage }) {
  const { prediction } = predictions;

  const defectReasons = {
    Crazing: {
      reason: 'Fine cracks on the surface forming a network pattern, typically caused by thermal stress during rapid cooling or heating cycles.',
      causes: ['Thermal stress from rapid temperature changes', 'Uneven cooling rates', 'Material fatigue', 'Improper heat treatment'],
      impact: 'Can lead to structural weakness and propagate into larger cracks over time.'
    },
    Inclusion: {
      reason: 'Non-metallic particles or impurities embedded in the metal matrix during manufacturing.',
      causes: ['Foreign particles in raw materials', 'Contamination during melting', 'Incomplete slag removal', 'Oxidation during processing'],
      impact: 'Creates stress concentration points that can initiate cracks and reduce material strength.'
    },
    Patches: {
      reason: 'Irregular surface areas with different texture or color, indicating surface contamination or processing irregularities.',
      causes: ['Scale formation during rolling', 'Surface oxidation', 'Uneven material composition', 'Contamination from rolling equipment'],
      impact: 'Affects surface finish quality and may indicate underlying material inconsistencies.'
    },
    Pitted: {
      reason: 'Small cavities or depressions on the surface caused by localized material loss.',
      causes: ['Corrosion from moisture or chemicals', 'Mechanical impact damage', 'Material porosity', 'Electrochemical reactions'],
      impact: 'Reduces load-bearing capacity and can serve as initiation sites for fatigue cracks.'
    },
    Rolled: {
      reason: 'Linear marks or indentations parallel to the rolling direction, caused by irregularities in the rolling mill.',
      causes: ['Debris on rolling mill rolls', 'Roll surface defects', 'Uneven pressure distribution', 'Material buildup on rolls'],
      impact: 'Compromises surface quality and dimensional accuracy of the rolled product.'
    },
    Scratches: {
      reason: 'Linear surface damage from mechanical contact or abrasion during handling or processing.',
      causes: ['Rough handling during transportation', 'Contact with sharp objects', 'Improper storage conditions', 'Equipment misalignment'],
      impact: 'Creates stress risers that may lead to crack initiation under load.'
    }
  };

  const getDefectInfo = (defectClass) => defectReasons[defectClass] || {
    reason: 'Defect information not available.',
    causes: [],
    impact: ''
  };

  const defectInfo = getDefectInfo(prediction.class);

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

      {/* Defect Analysis Panel */}
      <div className="defect-analysis-panel">
        <div className="analysis-header">
          <AlertCircle size={20} />
          <h3>Defect Analysis: {prediction.class}</h3>
        </div>

        <div className="analysis-section">
          <div className="analysis-label">
            <strong>Description:</strong>
          </div>
          <p className="analysis-text">{defectInfo.reason}</p>
        </div>

        {defectInfo.causes.length > 0 && (
          <div className="analysis-section">
            <div className="analysis-label">
              <strong>Possible Causes:</strong>
            </div>
            <ul className="causes-list">
              {defectInfo.causes.map((cause, index) => (
                <li key={index}>{cause}</li>
              ))}
            </ul>
          </div>
        )}

        {defectInfo.impact && (
          <div className="analysis-section impact-section">
            <div className="analysis-label">
              <strong>Impact on Quality:</strong>
            </div>
            <p className="analysis-text impact-text">{defectInfo.impact}</p>
          </div>
        )}
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
