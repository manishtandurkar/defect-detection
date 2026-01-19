"""
FastAPI Backend for Metallic Surface Defect Detection
with PaDiM-based Anomaly Localization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import base64
from typing import Dict, List
import cv2
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter
import pickle
import os

app = FastAPI(title="Defect Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "../defect_detection_model.pth"
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
IMG_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
model = None
padim_features = None
feature_extractor = None


class FeatureExtractor(nn.Module):
    """Extract features from intermediate layers for PaDiM"""
    def __init__(self, model):
        super().__init__()
        self.features = nn.ModuleList()
        
        # Extract features from ResNet layers
        if isinstance(model, models.ResNet):
            self.features.append(nn.Sequential(*list(model.children())[:5]))  # layer1
            self.features.append(nn.Sequential(*list(model.children())[5:6]))  # layer2
            self.features.append(nn.Sequential(*list(model.children())[6:7]))  # layer3
        
    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features


class PaDiM:
    """PaDiM-inspired: Activation-based Anomaly Localization"""
    def __init__(self, feature_extractor, device='cpu'):
        self.feature_extractor = feature_extractor
        self.device = device
        
    def extract_features(self, image_tensor):
        """Extract multi-scale features"""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        return features
    
    def compute_anomaly_map(self, image_tensor, target_size=(224, 224)):
        """Compute anomaly heatmap based on feature activation patterns"""
        features = self.extract_features(image_tensor)
        anomaly_maps = []
        
        for i, feat in enumerate(features):
            # Get feature activations
            feat_np = feat.cpu().numpy()
            b, c, h, w = feat_np.shape
            
            # Compute activation magnitudes across channels
            # Higher activations indicate areas the network is focusing on
            activation_map = np.mean(np.abs(feat_np[0]), axis=0)
            
            # Normalize to [0, 1]
            if activation_map.max() > activation_map.min():
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            
            # Compute variance across channels (higher variance = more diverse features = potential anomaly)
            variance_map = np.var(feat_np[0], axis=0)
            if variance_map.max() > variance_map.min():
                variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
            
            # Combine activation and variance
            combined_map = activation_map * 0.6 + variance_map * 0.4
            
            # Resize to target size
            combined_resized = cv2.resize(combined_map, target_size)
            anomaly_maps.append(combined_resized)
        
        # Weighted average (later layers have more semantic information)
        if len(anomaly_maps) >= 3:
            weights = [0.2, 0.3, 0.5]  # Give more weight to deeper layers
            final_map = sum(w * m for w, m in zip(weights, anomaly_maps))
        elif anomaly_maps:
            final_map = np.mean(anomaly_maps, axis=0)
        else:
            final_map = np.zeros(target_size)
        
        # Apply Gaussian smoothing for better visualization
        final_map = gaussian_filter(final_map, sigma=4)
        
        # Enhance contrast
        final_map = np.power(final_map, 0.8)  # Gamma correction
        
        # Normalize to [0, 100] for better score interpretation
        if final_map.max() > final_map.min():
            final_map = ((final_map - final_map.min()) / (final_map.max() - final_map.min())) * 100
        
        return final_map


def load_model():
    """Load the trained defect detection model"""
    global model, feature_extractor, padim_features
    
    try:
        # Load classification model
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Initialize feature extractor for PaDiM
        feature_extractor = FeatureExtractor(model).to(device)
        feature_extractor.eval()
        
        # Initialize PaDiM (reference features would be built from training data)
        padim_features = PaDiM(feature_extractor, device)
        
        print(f"✓ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


def preprocess_image(image: Image.Image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def create_heatmap_overlay(original_image: np.ndarray, anomaly_map: np.ndarray, alpha=0.5):
    """Create heatmap overlay on original image"""
    # Normalize anomaly map to [0, 255]
    anomaly_map_norm = ((anomaly_map - anomaly_map.min()) / 
                        (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize to match original image
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Overlay
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
    
    return overlay, heatmap


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    # Convert to PIL Image
    image = Image.fromarray(image_array.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Defect Detection API",
        "status": "running",
        "device": str(device),
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict defect class and generate PaDiM anomaly heatmap
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Preprocess for model
        image_tensor = preprocess_image(image)
        
        # Classification
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = float(confidence.item())
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy().tolist()
        class_probs = {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))}
        
        # Generate PaDiM anomaly map
        anomaly_map = padim_features.compute_anomaly_map(
            image_tensor, 
            target_size=(image_np.shape[0], image_np.shape[1])
        )
        
        # Create heatmap overlay
        overlay_image, heatmap_only = create_heatmap_overlay(image_np, anomaly_map, alpha=0.4)
        
        # Calculate anomaly score
        anomaly_score = float(np.max(anomaly_map))
        anomaly_mean = float(np.mean(anomaly_map))
        
        # Convert images to base64
        original_b64 = numpy_to_base64(image_np)
        overlay_b64 = numpy_to_base64(overlay_image)
        heatmap_b64 = numpy_to_base64(heatmap_only)
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": confidence_score,
                "all_probabilities": class_probs
            },
            "anomaly": {
                "score": anomaly_score,
                "mean_score": anomaly_mean,
                "interpretation": "Higher scores indicate more anomalous regions"
            },
            "images": {
                "original": original_b64,
                "overlay": overlay_b64,
                "heatmap": heatmap_b64
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "model": "ResNet50",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "device": str(device),
        "xai_method": "Feature Activation-based Anomaly Detection"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
