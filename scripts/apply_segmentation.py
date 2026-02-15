"""
Apply trained segmentation model to generate land cover maps

Generates predictions for all images in for_labeling directory.
Outputs colored visualization maps and raw prediction arrays.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Paths
PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
UNLABELED_DIR = PROJECT_DIR / 'data' / 'for_labeling'
CLASSES_FILE = PROJECT_DIR / 'data' / 'training_dataset' / 'classes.json'
MODEL_PATH = PROJECT_DIR / 'models' / 'segmentation_head.pth'
OUTPUT_DIR = PROJECT_DIR / 'results' / 'land_cover_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load classes
with open(CLASSES_FILE) as f:
    CLASSES = json.load(f)
NUM_CLASSES = len(CLASSES)

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# DINOv2 model
DINO_MODEL = "facebook/dinov2-large"
FEATURE_DIM = 1024

# Color mapping for visualization
CLASS_COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 255, 0],    # agriculture - yellow
    2: [0, 128, 0],      # forest - green
    3: [0, 0, 255],      # water - blue
    4: [255, 0, 0],      # urban - red
    5: [128, 0, 128],    # solar - purple
    6: [165, 42, 42]     # bare land - brown
}


class DINOFeatureExtractor(nn.Module):
    """DINOv2 feature extractor (same as training)"""

    def __init__(self, model_name=DINO_MODEL):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images):
        if images.max() <= 1.0:
            images = images * 255

        batch_size = images.shape[0]
        pil_images = []
        for i in range(batch_size):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token

        return features


class SegmentationHead(nn.Module):
    """Segmentation decoder head (same as training)"""
    
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, features, target_size):
        B, N, D = features.shape
        grid_size = int(N ** 0.5)
        features = features.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        logits = self.decoder(features)
        logits = F.interpolate(
            logits, size=target_size, mode='bilinear', align_corners=False
        )
        return logits


def predict_image(image_path, feature_extractor, seg_head):
    """
    Generate segmentation prediction for single image
    
    Args:
        image_path: Path to GeoTIFF or PNG image
        feature_extractor: DINOv3 feature extractor
        seg_head: Trained segmentation head
    
    Returns:
        prediction: (H, W) numpy array with class IDs
    """
    # Load image
    if image_path.suffix in ['.tif', '.tiff']:
        with rasterio.open(image_path) as src:
            image = src.read()
            if image.shape[0] > 3:
                image = image[:3]
    else:
        # PNG image
        img = Image.open(image_path)
        image = np.array(img).transpose(2, 0, 1) if len(np.array(img).shape) == 3 else np.array(img)
        if len(image.shape) == 2:
            image = np.stack([image, image, image])
    
    image = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        features = feature_extractor(image)
        logits = seg_head(features, image.shape[-2:])
        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()
    
    return prediction


def create_colored_map(prediction, class_colors):
    """
    Create RGB visualization from prediction array
    
    Args:
        prediction: (H, W) array with class IDs
        class_colors: dict mapping class_id -> [R, G, B]
    
    Returns:
        colored: (H, W, 3) RGB array
    """
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        mask = prediction == class_id
        colored[mask] = color
    
    return colored


if __name__ == '__main__':
    print("="*60)
    print("Applying Segmentation Model")
    print("="*60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Train model first with: python scripts/train_segmentation.py")
        exit(1)
    
    print("Loading models...")
    
    # Load models
    feature_extractor = DINOFeatureExtractor(DINO_MODEL)
    seg_head = SegmentationHead(FEATURE_DIM, NUM_CLASSES)
    seg_head.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    feature_extractor.to(DEVICE)
    seg_head.to(DEVICE)
    seg_head.eval()
    
    print(f"Models loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Find all images
    image_files = list(UNLABELED_DIR.glob('*.tif')) + list(UNLABELED_DIR.glob('*.tiff')) + list(UNLABELED_DIR.glob('*.png'))
    
    if not image_files:
        print(f"ERROR: No images found in {UNLABELED_DIR}")
        exit(1)
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        output_png = OUTPUT_DIR / f"{image_path.stem}_landcover.png"
        output_npy = OUTPUT_DIR / f"{image_path.stem}_prediction.npy"
        
        # Skip if already processed
        if output_png.exists() and output_npy.exists():
            continue
        
        # Predict
        prediction = predict_image(image_path, feature_extractor, seg_head)
        
        # Save raw prediction
        np.save(output_npy, prediction)
        
        # Create and save visualization
        colored_map = create_colored_map(prediction, CLASS_COLORS)
        Image.fromarray(colored_map).save(output_png)
    
    print(f"\n{'='*60}")
    print(f"Land cover maps saved to: {OUTPUT_DIR}")
    print(f"  - *_landcover.png: Colored visualization")
    print(f"  - *_prediction.npy: Raw class IDs for analysis")
    print(f"{'='*60}")

