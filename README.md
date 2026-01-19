# Metallic Surface Defect Detection Dashboard

A full-stack AI-powered defect detection system with Explainable AI (XAI) using PaDiM for anomaly localization.

## 🎯 Features

- **Real-time Defect Classification**: Classifies metallic surface defects into 6 categories
  - Crazing
  - Inclusion
  - Patches
  - Pitted
  - Rolled
  - Scratches

- **Explainable AI (XAI)**: PaDiM-based anomaly localization with interactive heatmaps
- **Interactive Dashboard**: Modern React UI with real-time visualization
- **Analysis History**: Track and review recent predictions
- **Statistics Panel**: Defect breakdown and analysis metrics

## 📁 Project Structure

```
.
├── backend/                    # FastAPI Backend
│   ├── app.py                 # Main API with PaDiM implementation
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── components/        # UI Components
│   │   │   ├── Header.jsx
│   │   │   ├── UploadSection.jsx
│   │   │   ├── ResultsSection.jsx
│   │   │   ├── XAISection.jsx
│   │   │   └── StatsPanel.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/                       # Dataset
│   ├── train/
│   ├── valid/
│   └── test/
│
├── defect_detection_model.pth  # Trained model (place here)
└── defect_detection_training.ipynb
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Trained model file: `defect_detection_model.pth`

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Important**: Place your trained model file in the project root:
```
Copy defect_detection_model.pth to: c:\Lab programs\AIML\EL\New method\
```

5. Start the backend server:
```bash
python app.py
```

The API will run on `http://localhost:8000`

### Frontend Setup

1. Open a new terminal and navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The dashboard will open at `http://localhost:3000`

## 🎮 Usage

1. **Upload Image**: Click or drag an image to the upload area
2. **View Results**: See the predicted defect class with confidence score
3. **Explore XAI**: Toggle between Original, Overlay, and Heatmap views
4. **Understand Anomalies**: Review the PaDiM anomaly scores and heatmap
5. **Check History**: View recent analyses in the sidebar

## 🔍 PaDiM Explanation

**PaDiM (Patch Distribution Modeling)** is an anomaly detection method that:

- Extracts multi-scale features from different layers of the neural network
- Builds a statistical distribution of "normal" patch features
- Computes Mahalanobis distance for test images
- Generates spatial anomaly heatmaps showing defect locations

**Heatmap Colors:**
- 🔵 **Blue/Purple**: Normal regions (low anomaly score)
- 🟡 **Yellow**: Moderate anomalies
- 🔴 **Red**: High anomalies (defect locations)

## 📊 API Endpoints

### `POST /predict`
Upload image for defect detection and anomaly localization

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Scratches",
    "confidence": 0.98,
    "all_probabilities": {...}
  },
  "anomaly": {
    "score": 85.2,
    "mean_score": 45.3,
    "interpretation": "..."
  },
  "images": {
    "original": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,..."
  }
}
```

### `GET /model_info`
Get model and configuration information

### `POST /build_padim_reference`
Build PaDiM reference distribution from training images (optional)

## 🛠️ Technologies Used

### Backend
- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- OpenCV - Image processing
- scikit-learn - Machine learning utilities
- SciPy - Scientific computing

### Frontend
- React 18 - UI framework
- Vite - Build tool
- Recharts - Chart visualization
- Lucide React - Icons
- Axios - HTTP client

## 🎨 Dashboard Features

### 1. Upload Section
- Drag & drop support
- Real-time preview
- Loading indicators

### 2. Results Section
- Detected defect class
- Confidence visualization
- Probability distribution chart

### 3. XAI Section
- Interactive view selector (Original/Overlay/Heatmap)
- Anomaly score metrics
- Detailed interpretation guide
- Color-coded severity levels

### 4. Statistics Panel
- Total analyses counter
- Defect breakdown chart
- Recent history with timestamps

## 🔧 Troubleshooting

### Backend Issues

**Model not found:**
```
Error: Model file not found at ../defect_detection_model.pth
```
→ Ensure the model file is in the correct location

**CUDA out of memory:**
→ The backend will automatically use CPU if CUDA is unavailable

### Frontend Issues

**CORS errors:**
→ Backend CORS is configured for localhost:3000. Update `app.py` if using different port

**API connection failed:**
→ Ensure backend is running on port 8000

## 📈 Model Performance

Based on training results:
- **Training Accuracy**: 99.94%
- **Validation Accuracy**: 100%
- **Test Accuracy**: 100%
- **Model**: ResNet50 with transfer learning
- **Classes**: 6 defect types

## 🚀 Production Deployment

### Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
# Build for production
npm run build

# Files will be in dist/ folder
# Deploy to any static hosting service
```

## 📝 Future Enhancements

- [ ] Batch image processing
- [ ] Export reports (PDF/CSV)
- [ ] User authentication
- [ ] Database integration for history
- [ ] Mobile responsive design improvements
- [ ] Real-time camera feed support

## 📄 License

This project is for educational purposes.

## 👥 Support

For issues or questions, please refer to the documentation or create an issue in the repository.

---

**Built with ❤️ for Quality Inspection**
