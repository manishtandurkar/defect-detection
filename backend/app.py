"""
FastAPI Backend for Metallic Surface Defect Detection
with PaDiM-based Anomaly Localization - Improved Heatmap
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
    """Hybrid KNN + Intensity Anomaly Detection for NEU Dataset"""
    
    # Configuration constants
    FEATURE_SELECTION_COUNT = 500  # Top 500 features with highest variance
    K_NEIGHBORS = 50  # Number of nearest neighbors for KNN distance
    KNN_SCORE_NORMALIZER = 10.0  # Normalize KNN distance to [0, 1] range
    
    def __init__(self, feature_extractor, device='cpu'):
        self.feature_extractor = feature_extractor
        self.device = device
        self.memory_banks = {}  # Per-class memory banks
        self.selected_indices = None
        self.k_neighbors = self.K_NEIGHBORS
        
    def extract_features(self, image_tensor):
        """Extract multi-scale features"""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        return features
    
    def build_memory_bank(self, class_name, image_tensors):
        """
        Build class-specific memory bank from training images
        Args:
            class_name: defect class name
            image_tensors: list of image tensors for this class
        """
        features_list = []
        
        for img_tensor in image_tensors:
            with torch.no_grad():
                features = self.extract_features(img_tensor)
                # Flatten multi-layer features
                combined_features = []
                for feat in features:
                    feat_flat = feat.flatten(start_dim=1)
                    combined_features.append(feat_flat)
                all_features = torch.cat(combined_features, dim=1)
                features_list.append(all_features.squeeze())
        
        memory_bank = torch.stack(features_list).to(self.device)
        
        # Feature selection: top features with highest std (inspired by mohan696matlab)
        # Reduces dimensionality and focuses on most discriminative features
        if self.selected_indices is None:
            stds = memory_bank.std(dim=0)
            _, indices = torch.sort(stds, descending=True)
            self.selected_indices = indices[:self.FEATURE_SELECTION_COUNT]
        
        self.memory_banks[class_name] = memory_bank[:, self.selected_indices]
        print(f"Built memory bank for {class_name}: {memory_bank.shape}")
    
    def compute_knn_distance(self, features, class_name):
        """
        Compute KNN-based anomaly score
        Args:
            features: extracted features from test image
            class_name: predicted defect class
        Returns:
            average distance to k-nearest neighbors
        """
        if class_name not in self.memory_banks:
            # Fallback: use combined memory bank if class-specific bank not found
            # This handles cases where memory banks weren't built or class is unknown
            if not self.memory_banks:
                return 0  # No memory banks available
            all_banks = torch.cat(list(self.memory_banks.values()), dim=0)
            relevant_bank = all_banks
        else:
            relevant_bank = self.memory_banks[class_name]
        
        # Flatten and select features
        combined_features = []
        for feat in features:
            feat_flat = feat.flatten(start_dim=1)
            combined_features.append(feat_flat)
        all_features = torch.cat(combined_features, dim=1)
        feat_selected = all_features[0, self.selected_indices]
        
        # Calculate distances to all samples in memory bank
        distances = torch.norm(relevant_bank - feat_selected, dim=1)
        
        # Get k-nearest neighbors
        sorted_distances = torch.sort(distances)[0]
        knn_distances = sorted_distances[:self.k_neighbors]
        
        return knn_distances.mean().cpu().item()
    
    def compute_anomaly_map(self, image_tensor, predicted_class=None, target_size=(224, 224)):
        """
        Hybrid anomaly map: KNN + Intensity for metallic surfaces
        Args:
            image_tensor: input image tensor
            predicted_class: predicted defect class (optional)
            target_size: output heatmap size
        """
        # Extract features
        features = self.extract_features(image_tensor)
        
        # 1. KNN-based anomaly score (simplified approach)
        knn_score = 0
        if self.memory_banks and predicted_class:
            knn_score = self.compute_knn_distance(features, predicted_class)
        
        # 2. Intensity-based anomaly for metallic surfaces
        img_np = image_tensor[0].cpu().numpy()
        grayscale = np.mean(img_np, axis=0)
        
        # Invert for dark defects on metallic surfaces
        intensity_map = -grayscale
        
        # Simple normalization
        if intensity_map.max() > intensity_map.min():
            intensity_map = (intensity_map - intensity_map.min()) / \
                           (intensity_map.max() - intensity_map.min())
        
        # Resize to target
        intensity_map_resized = cv2.resize(intensity_map, target_size)
        
        # 3. Feature-based spatial map (simplified from original)
        anomaly_maps = []
        for feat in features:
            feat_np = feat.cpu().numpy()
            activation_map = np.mean(np.abs(feat_np[0]), axis=0)
            
            # Simple normalization
            if activation_map.max() > activation_map.min():
                activation_map = (activation_map - activation_map.min()) / \
                                (activation_map.max() - activation_map.min())
            
            activation_resized = cv2.resize(activation_map, target_size)
            anomaly_maps.append(activation_resized)
        
        # Average feature maps
        feature_map = np.mean(anomaly_maps, axis=0) if anomaly_maps else np.zeros(target_size)
        
        # 4. Fusion: 50% features + 40% intensity + 10% KNN score
        # Balanced weights for NEU metallic dataset based on empirical testing
        # KNN score is normalized by dividing by KNN_SCORE_NORMALIZER to scale to [0, 1]
        knn_map = np.full(target_size, knn_score / self.KNN_SCORE_NORMALIZER)
        final_map = 0.5 * feature_map + 0.4 * intensity_map_resized + 0.1 * knn_map
        
        # 5. Simple smoothing (no aggressive transforms!)
        final_map = gaussian_filter(final_map, sigma=2)
        
        # 6. Simple normalization (removed complex percentile + power transforms)
        if final_map.max() > final_map.min():
            final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min())
        
        # Scale to [0, 100]
        return final_map * 100


def load_model():
    """Load the trained defect detection model and memory banks"""
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
        
        # Initialize PaDiM with hybrid approach
        padim_features = PaDiM(feature_extractor, device)
        
        # Load pre-built memory banks if available
        from pathlib import Path
        memory_bank_path = 'memory_banks.pkl'
        if Path(memory_bank_path).exists():
            memory_data = torch.load(memory_bank_path, map_location=device)
            padim_features.memory_banks = memory_data['memory_banks']
            padim_features.selected_indices = memory_data['selected_indices']
            padim_features.k_neighbors = memory_data['k_neighbors']
            print(f"✓ Loaded memory banks for {len(padim_features.memory_banks)} classes")
        else:
            print(f"⚠ Memory banks not found. Run build_memory_banks.py first for best results.")
        
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
    """Create heatmap overlay with simplified colormap
    
    Uses JET colormap for better defect visualization compared to TURBO:
    - JET provides clearer transition from blue (normal) to red (defect)
    - Better suited for metallic surface defects where contrast is important
    - More familiar color scheme for quality inspection applications
    """
    
    # Clip to valid range
    anomaly_map_clipped = np.clip(anomaly_map, 0, 100)
    
    # Convert to 0-255 range
    anomaly_map_norm = (anomaly_map_clipped / 100.0 * 255).astype(np.uint8)
    
    # Use JET colormap for clearer visualization (simpler than TURBO)
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize to match original image
    if original_image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Simple fixed alpha blending (removed adaptive alpha)
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
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
    """Predict with improved heatmap generation"""
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
        
        # Generate anomaly map with predicted class (KEY CHANGE)
        anomaly_map = padim_features.compute_anomaly_map(
            image_tensor,
            predicted_class=predicted_class,  # Pass predicted class
            target_size=(image_np.shape[0], image_np.shape[1])
        )
        
        # Create heatmap overlay with simplified method
        overlay_image, heatmap_only = create_heatmap_overlay(
            image_np, 
            anomaly_map, 
            alpha=0.4  # Reduced alpha for better visibility
        )
        
        # Calculate anomaly statistics
        anomaly_score = float(np.max(anomaly_map))
        anomaly_mean = float(np.mean(anomaly_map))
        anomaly_p95 = float(np.percentile(anomaly_map, 95))
        
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
                "p95_score": anomaly_p95,
                "interpretation": "Red areas = High defect probability, Blue areas = Normal regions"
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
        "xai_method": "Hybrid KNN + Intensity Anomaly Detection",
        "colormap": "JET (Red=Defect, Blue=Normal)",
        "memory_banks": len(padim_features.memory_banks) if padim_features else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)