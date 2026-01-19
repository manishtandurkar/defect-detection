import React, { useState } from 'react';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import XAISection from './components/XAISection';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedImage(URL.createObjectURL(file));
    setSelectedFile(file);
    setPredictions(null);
    setError(null);
  };

  const handleProcess = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data);
      
    } catch (err) {
      setError(err.message);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setPredictions(null);
    setError(null);
  };

  return (
    <div className="app">
      <Header />
      
      <div className="container">
        <UploadSection 
          onImageSelect={handleImageSelect}
          onProcess={handleProcess}
          loading={loading}
          onReset={handleReset}
          selectedImage={selectedImage}
          hasResults={!!predictions}
        />
        
        {error && (
          <div className="error-banner">
            <span>❌ Error: {error}</span>
          </div>
        )}
        
        {predictions && (
          <>
            <ResultsSection 
              predictions={predictions}
              originalImage={selectedImage}
            />
            
            <XAISection 
              images={predictions.images}
              anomaly={predictions.anomaly}
              prediction={predictions.prediction}
            />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
