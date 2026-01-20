"""
Utility script to build per-class memory banks for improved anomaly detection
Run this after training the classification model

Usage:
    python build_memory_banks.py --train_path ../data/NEU/train --model_path ../defect_detection_model.pth
"""

import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import argparse
import sys

# Configuration
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
IMG_SIZE = 224
FEATURE_SELECTION_COUNT = 500  # Top 500 features with highest std
K_NEIGHBORS = 50

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


class MemoryBankBuilder:
    """Build memory banks for KNN-based anomaly detection"""
    def __init__(self, feature_extractor, device='cpu'):
        self.feature_extractor = feature_extractor
        self.device = device
        self.memory_banks = {}
        self.selected_indices = None
        
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
        
        # Feature selection: top features with highest std (like mohan696matlab)
        if self.selected_indices is None:
            stds = memory_bank.std(dim=0)
            _, indices = torch.sort(stds, descending=True)
            self.selected_indices = indices[:FEATURE_SELECTION_COUNT]
        
        self.memory_banks[class_name] = memory_bank[:, self.selected_indices]
        print(f"Built memory bank for {class_name}: {memory_bank.shape}")


def load_model(model_path, device):
    """Load the trained defect detection model"""
    try:
        # Load classification model
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def build_all_memory_banks(train_data_path, model_path, output_path='memory_banks.pkl'):
    """Build memory banks for all defect classes"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Initialize memory bank builder
    builder = MemoryBankBuilder(feature_extractor, device)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_path = Path(train_data_path)
    
    if not train_path.exists():
        print(f"Error: Training data path does not exist: {train_path}")
        sys.exit(1)
    
    for class_name in CLASS_NAMES:
        class_folder = train_path / class_name
        if not class_folder.exists():
            print(f"Warning: {class_folder} does not exist")
            continue
        
        print(f"Building memory bank for {class_name}...")
        
        # Load all training images for this class
        image_tensors = []
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        if not image_files:
            print(f"Warning: No images found in {class_folder}")
            continue
            
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                image_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if image_tensors:
            builder.build_memory_bank(class_name, image_tensors)
            print(f"  Loaded {len(image_tensors)} images")
        else:
            print(f"No images found for {class_name}")
    
    # Save memory banks
    torch.save({
        'memory_banks': builder.memory_banks,
        'selected_indices': builder.selected_indices,
        'k_neighbors': K_NEIGHBORS,
        'feature_selection_count': FEATURE_SELECTION_COUNT
    }, output_path)
    
    print(f"\n✓ Memory banks saved to {output_path}")
    print(f"  Total classes: {len(builder.memory_banks)}")
    print(f"  Feature dimension: {FEATURE_SELECTION_COUNT}")
    print(f"  K-neighbors: {K_NEIGHBORS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build memory banks for defect detection')
    parser.add_argument('--train_path', type=str, default='../data/NEU/train',
                      help='Path to training data directory')
    parser.add_argument('--model_path', type=str, default='../defect_detection_model.pth',
                      help='Path to trained model file')
    parser.add_argument('--output_path', type=str, default='memory_banks.pkl',
                      help='Path to save memory banks')
    
    args = parser.parse_args()
    
    build_all_memory_banks(args.train_path, args.model_path, args.output_path)

