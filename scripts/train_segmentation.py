"""
Train DINOv3 segmentation model for land cover classification

Uses DINOv3 satellite model (facebook/dinov3-vitl16-pretrain-sat493m) as frozen
feature extractor, trains lightweight decoder head for segmentation.

Model auto-downloads on first run (~1.2GB).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Paths
PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
IMAGES_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'images'
MASKS_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
CLASSES_FILE = PROJECT_DIR / 'data' / 'training_dataset' / 'classes.json'
MODEL_DIR = PROJECT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load classes
with open(CLASSES_FILE) as f:
    CLASSES = json.load(f)
NUM_CLASSES = len(CLASSES)

# Hyperparameters
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 0.001

# Device selection
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# DINOv2 model configuration (freely available, 1024-dim features)
DINO_MODEL = "facebook/dinov2-large"
FEATURE_DIM = 1024

print(f"Device: {DEVICE}")
print(f"Classes: {NUM_CLASSES}")
print(f"Class mapping: {CLASSES}")


class LandCoverDataset(Dataset):
    """
    Dataset for land cover segmentation
    
    Expected structure:
    - images/mongla_5km_2023.tif
    - masks/mongla_5km_2023_mask.png
    
    Masks should be PNG files with pixel values corresponding to class IDs
    (0=background, 1=agriculture, 2=forest, etc.)
    """
    
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Find all image files (GeoTIFF and PNG)
        image_files = (list(self.images_dir.glob('*.tif'))
                       + list(self.images_dir.glob('*.tiff'))
                       + list(self.images_dir.glob('*.png')))
        
        # Filter to only images with corresponding masks
        self.samples = []
        for img_path in image_files:
            mask_path = self.masks_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                self.samples.append({'image': img_path, 'mask': mask_path})
            else:
                print(f"Warning: No mask found for {img_path.name}")
        
        print(f"Found {len(self.samples)} image-mask pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (GeoTIFF or PNG)
        if sample['image'].suffix in ('.tif', '.tiff'):
            with rasterio.open(sample['image']) as src:
                image = src.read()  # (C, H, W)
                if image.shape[0] > 3:
                    image = image[:3]
        else:
            img = Image.open(sample['image']).convert('RGB')
            image = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float()
        
        # Load mask (PNG)
        mask = np.array(Image.open(sample['mask']))
        # Convert to single channel if RGB
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'filename': sample['image'].name
        }


class DINOFeatureExtractor(nn.Module):
    """
    Extract features from DINOv2-large model.

    Model is frozen during training. Only the segmentation head is trained.
    Downloads automatically on first instantiation (~1.2GB).
    """

    def __init__(self, model_name=DINO_MODEL):
        super().__init__()

        print(f"Loading DINOv2 model: {model_name}")
        print("(First run will download ~1.2GB model)")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        print("DINOv2 loaded and frozen")

    def forward(self, images):
        """
        Args:
            images: (B, C, H, W) tensor, values in [0, 1] or [0, 255]
        Returns:
            features: (B, num_patches, feature_dim) tensor
        """
        if images.max() <= 1.0:
            images = images * 255

        # Convert to PIL for processor
        batch_size = images.shape[0]
        pil_images = []
        for i in range(batch_size):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Process
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # DINOv2 includes CLS token at position 0 — remove it
            features = outputs.last_hidden_state[:, 1:, :]  # (B, num_patches, feature_dim)

        return features


class SegmentationHead(nn.Module):
    """
    Decoder head: patch features -> full resolution segmentation map
    
    Upsamples DINOv3 patch features to match input image resolution.
    """
    
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Upsampling decoder
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
        """
        Args:
            features: (B, num_patches, feature_dim)
            target_size: (H, W) tuple
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, N, D = features.shape
        
        # Reshape to spatial grid (assuming square patches)
        grid_size = int(N ** 0.5)
        features = features.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        
        # Decode
        logits = self.decoder(features)
        
        # Final upsample to target size
        logits = F.interpolate(
            logits, size=target_size, mode='bilinear', align_corners=False
        )
        
        return logits


def train_model(train_loader, feature_extractor, seg_head, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Train segmentation head on frozen DINOv3 features
    """
    
    feature_extractor.to(DEVICE)
    seg_head.to(DEVICE)
    
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining on {DEVICE}")
    print(f"Epochs: {epochs}, Learning rate: {lr}\n")
    
    for epoch in range(epochs):
        seg_head.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            # Extract features (frozen backbone)
            features = feature_extractor(images)
            
            # Predict segmentation
            logits = seg_head(features, masks.shape[-2:])
            
            # Compute loss
            loss = criterion(logits, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    return seg_head


if __name__ == '__main__':
    print("="*60)
    print("DINOv3 Land Cover Segmentation Training")
    print("="*60)
    
    # Create dataset
    dataset = LandCoverDataset(IMAGES_DIR, MASKS_DIR)
    
    if len(dataset) == 0:
        print("ERROR: No training samples found!")
        print(f"Check that images exist in: {IMAGES_DIR}")
        print(f"Check that masks exist in: {MASKS_DIR}")
        print("Expected mask naming: {image_stem}_mask.png")
        exit(1)
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    feature_extractor = DINOFeatureExtractor()
    seg_head = SegmentationHead(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    
    # Train
    trained_head = train_model(train_loader, feature_extractor, seg_head)
    
    # Save model
    model_path = MODEL_DIR / 'segmentation_head.pth'
    torch.save(trained_head.state_dict(), model_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}")

