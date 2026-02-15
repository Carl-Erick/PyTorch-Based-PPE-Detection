"""
Preprocessing Pipeline for PPE Detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from torchvision import transforms


class PreprocessingPipeline:
    """
    Preprocessing utilities for PPE detection images and datasets.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (640, 640)):
        """
        Initialize PreprocessingPipeline.
        
        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def resize_image(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to specified size.
        
        Args:
            image: Input image
            size: Target size (height, width), defaults to self.image_size
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.image_size
        return cv2.resize(image, (size[1], size[0]))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image from BGR to RGB format.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Image in RGB format
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        normalize: bool = True,
        return_tensor: bool = False
    ) -> np.ndarray or torch.Tensor:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image: Input image
            normalize: Whether to normalize pixel values
            return_tensor: Whether to return as PyTorch tensor
            
        Returns:
            Preprocessed image
        """
        # Resize
        image = self.resize_image(image)
        
        # Convert BGR to RGB
        image = self.convert_bgr_to_rgb(image)
        
        # Normalize
        if normalize:
            image = self.normalize_image(image)
        
        # Convert to tensor
        if return_tensor:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            image_tensor = self.normalize(image_tensor)
            return image_tensor
        
        return image
    
    def load_bbox_annotations(self, annotation_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load bounding box annotations from file (YOLO format).
        
        Args:
            annotation_path: Path to annotation file
            
        Returns:
            Tuple of (bboxes, labels)
        """
        bboxes = []
        labels = []
        
        try:
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:])
                            
                            # Convert to [x_min, y_min, x_max, y_max]
                            x_min = (x_center - w/2)
                            y_min = (y_center - h/2)
                            x_max = (x_center + w/2)
                            y_max = (y_center + h/2)
                            
                            bboxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
        except FileNotFoundError:
            print(f"Annotation file not found: {annotation_path}")
        
        return np.array(bboxes), np.array(labels)
    
    def scale_bboxes(
        self,
        bboxes: np.ndarray,
        original_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Scale bounding boxes to new image size.
        
        Args:
            bboxes: Bounding boxes in normalized format
            original_size: Original image size (height, width)
            target_size: Target image size (height, width)
            
        Returns:
            Scaled bounding boxes
        """
        if target_size is None:
            target_size = self.image_size
        
        scale_x = target_size[1] / original_size[1]
        scale_y = target_size[0] / original_size[0]
        
        scaled_bboxes = bboxes.copy()
        scaled_bboxes[:, [0, 2]] *= scale_x
        scaled_bboxes[:, [1, 3]] *= scale_y
        
        return scaled_bboxes
    
    def clip_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Clip bounding boxes to valid image coordinates.
        
        Args:
            bboxes: Bounding boxes
            
        Returns:
            Clipped bounding boxes
        """
        clipped = bboxes.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, self.image_size[1])
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, self.image_size[0])
        return clipped
