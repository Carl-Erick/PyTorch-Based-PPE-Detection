"""
Data Augmentation Module for PPE Detection
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict
import random


class DataAugmenter:
    """
    Data augmentation utilities for PPE detection dataset.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (640, 640)):
        """
        Initialize DataAugmenter.
        
        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
        self.transform_train = self._get_train_transforms()
        self.transform_val = self._get_val_transforms()
    
    def _get_train_transforms(self) -> A.Compose:
        """
        Get training augmentation pipeline.
        
        Returns:
            Albumentations Compose object with augmentations
        """
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def _get_val_transforms(self) -> A.Compose:
        """
        Get validation augmentation pipeline (minimal).
        
        Returns:
            Albumentations Compose object with minimal augmentations
        """
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def augment_image(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, mode: str = 'train'):
        """
        Apply augmentation to an image and its bounding boxes.
        
        Args:
            image: Input image array (H, W, 3)
            bboxes: Bounding boxes in format [x_min, y_min, x_max, y_max]
            labels: Class labels for each bounding box
            mode: Either 'train' or 'val'
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_labels)
        """
        transform = self.transform_train if mode == 'train' else self.transform_val
        
        if len(bboxes) > 0:
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            return augmented['image'], augmented['bboxes'], augmented['labels']
        else:
            augmented = transform(image=image, bboxes=[], labels=[])
            return augmented['image'], augmented['bboxes'], augmented['labels']
    
    def apply_custom_augmentation(
        self,
        image: np.ndarray,
        augmentation_type: str = 'random'
    ) -> np.ndarray:
        """
        Apply custom augmentation to an image.
        
        Args:
            image: Input image array
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'brightness':
            return cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        elif augmentation_type == 'blur':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif augmentation_type == 'flip':
            return cv2.flip(image, 1)
        elif augmentation_type == 'rotation':
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        elif augmentation_type == 'noise':
            noise = np.random.normal(0, 25, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        else:
            # Random augmentation
            aug_types = ['brightness', 'blur', 'flip', 'noise']
            return self.apply_custom_augmentation(image, random.choice(aug_types))
