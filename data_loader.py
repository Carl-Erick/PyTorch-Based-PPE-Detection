"""
Simple PPE Detection Data Loading Module
Focuses on loading and displaying PPE images
PyTorch-compatible loader
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms


class PPEDataLoader:
    """Simple data loader for PPE images"""
    
    def __init__(self, images_dir: str):
        """
        Initialize data loader
        
        Args:
            images_dir: Path to images directory
        """
        self.images_dir = Path(images_dir)
        self.image_files = []
        self.load_images()
    
    def load_images(self):
        """Load list of images from directory"""
        if self.images_dir.exists():
            self.image_files = sorted([
                f for f in os.listdir(self.images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        print(f"Loaded {len(self.image_files)} images")
    
    def get_image(self, idx: int) -> np.ndarray:
        """
        Get image by index
        
        Args:
            idx: Image index
            
        Returns:
            Image as numpy array
        """
        image_path = self.images_dir / self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def get_all_images(self) -> List[np.ndarray]:
        """Get all images"""
        images = []
        for idx in range(len(self.image_files)):
            images.append(self.get_image(idx))
        return images
    
    def get_image_stats(self) -> dict:
        """Get statistics about images"""
        stats = {
            'total_images': len(self.image_files),
            'image_names': self.image_files
        }
        return stats
    
    def get_image_tensor(self, idx: int) -> torch.Tensor:
        """
        Get image as PyTorch tensor
        
        Args:
            idx: Image index
            
        Returns:
            Tensor of shape (3, H, W) with values [0, 1]
        """
        image = self.get_image(idx)
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        # Convert to tensor and permute to (C, H, W)
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        return tensor
