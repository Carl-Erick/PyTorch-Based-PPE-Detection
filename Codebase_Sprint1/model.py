"""
Simple PyTorch-based PPE Detection Model
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimplePPEModel(nn.Module):
    """Simple CNN model for PPE classification"""
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize model
        
        Args:
            num_classes: Number of PPE classes (helmet, vest, no_ppe)
        """
        super(SimplePPEModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
            
        Returns:
            Output logits
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class PPEModelInference:
    """Simple inference wrapper for PPE model"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize inference
        
        Args:
            model_path: Path to saved model
            device: Device to use (cpu/cuda)
        """
        self.device = device
        self.model = SimplePPEModel(num_classes=3).to(device)
        self.class_names = ['Helmet', 'Vest', 'No PPE']
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predict PPE class for image tensor
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Tuple of (class_name, confidence)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return self.class_names[pred_class], confidence
