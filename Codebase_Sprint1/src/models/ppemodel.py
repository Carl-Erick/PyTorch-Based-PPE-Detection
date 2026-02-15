"""
PPE Detection Model using Transfer Learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional


class PPEDetectionModel(nn.Module):
    """
    PPE Detection Model using ResNet backbone with FPN.
    Suitable for object detection tasks.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_features: int = 256
    ):
        """
        Initialize PPEDetectionModel.
        
        Args:
            num_classes: Number of PPE classes (e.g., helmet, vest, no_ppe)
            backbone: Backbone network type (resnet50, resnet101, etc.)
            pretrained: Whether to use pretrained weights
            num_features: Number of features in FPN layers
        """
        super(PPEDetectionModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.num_features = num_features
        
        # Load backbone
        if backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            backbone_model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        
        # Feature Pyramid Network
        self.fpn = self._build_fpn()
        
        # Detection heads
        self.classification_head = self._build_classification_head()
        self.regression_head = self._build_regression_head()
    
    def _build_fpn(self) -> nn.Module:
        """Build Feature Pyramid Network."""
        return nn.Sequential(
            nn.Conv2d(2048, self.num_features, kernel_size=1),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True)
        )
    
    def _build_classification_head(self) -> nn.Sequential:
        """Build classification head for PPE detection."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
    
    def _build_regression_head(self) -> nn.Sequential:
        """Build regression head for bounding box prediction."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # 4 coordinates for bounding box
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (class_logits, bbox_predictions)
        """
        # Backbone
        features = self.backbone(x)
        
        # FPN
        features = self.fpn(features)
        
        # Detection heads
        class_logits = self.classification_head(features)
        bbox_pred = self.regression_head(features)
        
        return class_logits, bbox_pred


class SimplePPEClassifier(nn.Module):
    """
    Simple CNN-based PPE classifier for image classification.
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize SimplePPEClassifier.
        
        Args:
            num_classes: Number of PPE classes
        """
        super(SimplePPEClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
