import React, { useRef } from 'react';
import { Upload, Loader, RefreshCw, Play } from 'lucide-react';
import './UploadSection.css';

function UploadSection({ onImageSelect, onProcess, loading, onReset, selectedImage, hasResults }) {
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelect(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="upload-section card">
      <h2>Upload Image for Analysis</h2>
      
      {!selectedImage ? (
        <div 
          className="upload-area"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <Upload size={48} />
          <p>Click or drag image here</p>
          <span className="hint">Supports: JPG, PNG, JPEG</span>
        </div>
      ) : (
        <div className="image-preview-section">
          <div className="preview-container">
            <img src={selectedImage} alt="Selected" className="preview-image" />
          </div>
          
          <div className="action-buttons">
            {!hasResults && (
              <button 
                className="btn-process" 
                onClick={onProcess}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader className="spin-icon" size={20} />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Process Image
                  </>
                )}
              </button>
            )}
            
            <button className="btn-new" onClick={onReset}>
              <RefreshCw size={18} />
              Add New Image
            </button>
          </div>
        </div>
      )}
      
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default UploadSection;
