"""
Data Loading Module for PPE Detection Dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class PPEDataset(Dataset):
    """
    Custom Dataset class for PPE Detection.
    
    Expected directory structure:
    dataset/
        images/
            image1.jpg
            image2.jpg
            ...
        annotations/
            image1.txt
            image2.txt
            ...
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        transforms: Optional[object] = None,
        ppe_classes: List[str] = None
    ):
        """
        Initialize the PPE Dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_dir: Path to directory containing annotations
            transforms: Image transformations to apply
            ppe_classes: List of PPE class names
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms
        self.ppe_classes = ppe_classes or ['helmet', 'vest', 'no_ppe']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.ppe_classes)}
        
        # Get list of images
        self.image_paths = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, targets) where targets is a dict with labels and boxes
        """
        image_path = self.image_dir / self.image_paths[idx]
        annotation_path = self.annotation_dir / self.image_paths[idx].replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations (YOLO format: class_id x_center y_center width height)
        boxes = []
        labels = []
        
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:])
                            
                            # Convert to [x_min, y_min, x_max, y_max]
                            img_width, img_height = image.size
                            x_min = (x_center - w/2) * img_width
                            y_min = (y_center - h/2) * img_height
                            x_max = (x_center + w/2) * img_width
                            y_max = (y_center + h/2) * img_height
                            
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        targets = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        return image, targets


class PPEDataLoader:
    """
    Utility class for creating data loaders for PPE detection.
    """
    
    @staticmethod
    def get_data_loader(
        image_dir: str,
        annotation_dir: str,
        batch_size: int = 32,
        train_transforms: Optional[object] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for PPE detection dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_dir: Path to directory containing annotations
            batch_size: Batch size for DataLoader
            train_transforms: Image transformations
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader object
        """
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        dataset = PPEDataset(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            transforms=train_transforms
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )
        
        return dataloader
    
    @staticmethod
    def collate_fn(batch: List) -> Tuple[torch.Tensor, List]:
        """
        Custom collate function for objects detection.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batch of images and targets
        """
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, dim=0)
        return images, targets
