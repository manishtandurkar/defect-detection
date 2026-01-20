"""
Utility script to build per-class memory banks for improved anomaly detection
Run this after training the classification model
"""

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pickle

# Import from app.py
from app import model, feature_extractor, padim_features, CLASS_NAMES, IMG_SIZE, device

def build_all_memory_banks(train_data_path='path/to/NEU/train'):
    """Build memory banks for all defect classes"""
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_path = Path(train_data_path)
    
    for class_name in CLASS_NAMES:
        class_folder = train_path / class_name
        if not class_folder.exists():
            print(f"Warning: {class_folder} does not exist")
            continue
        
        print(f"Building memory bank for {class_name}...")
        
        # Load all training images for this class
        image_tensors = []
        for img_path in class_folder.glob('*.jpg'):  # Adjust extension if needed
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                image_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if image_tensors:
            padim_features.build_memory_bank(class_name, image_tensors)
        else:
            print(f"No images found for {class_name}")
    
    # Save memory banks
    save_path = 'memory_banks.pkl'
    torch.save({
        'memory_banks': padim_features.memory_banks,
        'selected_indices': padim_features.selected_indices,
        'k_neighbors': padim_features.k_neighbors
    }, save_path)
    
    print(f"Memory banks saved to {save_path}")

if __name__ == "__main__":
    # Update this path to your NEU training data
    build_all_memory_banks(train_data_path='../data/NEU/train')
