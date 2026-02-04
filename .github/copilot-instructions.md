# Copilot Instructions (Metallic Surface Defect Detection)

## Big picture architecture
- Three-tier app: FastAPI backend in backend/, React + Vite frontend in frontend/, model artifacts at repo root (defect_detection_model.pth).
- Backend uses ResNet50 classifier (6 defect classes) + hybrid PaDiM anomaly detector with 3 methods: feature-based (60%), intensity-based (25%), KNN-based (15%).
- PaDiM extracts features from 3 ResNet layers (layer1/2/3 → 56×56, 28×28, 14×14), builds per-class memory banks, generates JET colormap heatmaps.
- Frontend App.jsx maintains state (selectedImage, predictions), POSTs FormData to /predict, displays results via ResultsSection (classification + bar chart) and XAISection (heatmap overlay).

## Critical workflows
**Backend setup & run:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows; use source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python main.py  # dev server with auto-reload on port 8000
```
- Direct run: `python app.py` (no reload).
- Model loads on startup (@app.on_event("startup")) from MODEL_PATH="../defect_detection_model.pth".
- Optional: build memory banks with `python build_memory_banks.py --train_path ../data/train --model_path ../defect_detection_model.pth` → saves backend/memory_banks.pkl.

**Frontend setup & run:**
```bash
cd frontend
npm install
npm run dev  # Vite dev server on port 5173
```

## Project-specific conventions & data flow
**Fixed class names (order matters for model output):** Crazing, Inclusion, Patches, Pitted, Rolled, Scratches.

**Image processing pipeline:**
1. Frontend: user uploads image → File object stored in selectedFile state
2. Backend /predict: receives multipart/form-data, converts to PIL RGB, preprocesses to 224×224 tensor with ImageNet normalization
3. Classification: ResNet50 forward pass → 6-class logits → softmax probabilities
4. Anomaly detection: FeatureExtractor pulls intermediate activations → PaDiM.compute_anomaly_map() fuses 3 scores
5. Heatmap generation: JET colormap (blue=normal, red=defect) with 40% alpha blending (cv2.addWeighted)
6. Response: JSON with prediction (class, confidence, all_probabilities), anomaly (score, mean_score), images (original/heatmap as base64)

**Memory banks (backend/memory_banks.pkl):**
- Per-class feature tensors from training images; enables KNN distance calculation.
- If missing, PaDiM still works but KNN contribution is zero (falls back to features + intensity only).
- Rebuild after model retraining or when adding new training samples.

**Key constants in backend/app.py:**
- FEATURE_SELECTION_COUNT=500, K_NEIGHBORS=50, VARIANCE_FILTER_SIZE=5, INTENSITY_PERCENTILE_THRESHOLD=60.
- Fusion weights: 0.60*feature_map + 0.25*intensity_map + 0.15*knn_map (line ~270).

## Integration points & API contract
**CORS:** backend allows http://localhost:3000 and http://localhost:5173 (line ~24 in app.py).

**POST /predict response shape:**
```json
{
  "success": true,
  "prediction": {
    "class": "Scratches",
    "confidence": 0.98,
    "all_probabilities": {"Crazing": 0.01, "Inclusion": 0.005, ...}
  },
  "anomaly": {"score": 85.2, "mean_score": 45.3, "interpretation": "..."},
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,..."
  }
}
```
- Frontend expects `images.original` and `images.heatmap` (overlay not used in current UI).

## Common gotchas
- **Model not found error:** ensure defect_detection_model.pth exists at repo root (not in backend/).
- **CORS errors:** if frontend runs on different port, update allow_origins in backend/app.py line 24.
- **Memory banks warning:** "⚠ Memory banks not found" is non-fatal; KNN detection disabled but app works.
- **Image preprocessing:** backend expects RGB images; grayscale converted via .convert('RGB').
- **Port conflicts:** backend default 8000, frontend default 5173; check terminal output for actual ports.

## Key files for AI agents
- **backend/app.py:** FastAPI routes, PaDiM class (lines 78-285), FeatureExtractor (lines 47-62), heatmap generation (lines 371-400).
- **backend/build_memory_banks.py:** Standalone script to precompute per-class feature banks from data/train.
- **frontend/src/App.jsx:** Root component with handleProcess (POST /predict), state management (lines 5-14).
- **frontend/src/components/UploadSection.jsx:** Drag-and-drop file upload with preview.
- **frontend/src/components/ResultsSection.jsx:** Displays predicted class, confidence bar, probability chart (Recharts), defect analysis panel with causes/impacts.
- **frontend/src/components/XAISection.jsx:** Renders anomaly heatmap, severity badges, color legend, interpretation guide.
