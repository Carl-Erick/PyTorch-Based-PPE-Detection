"""
YOLOv3 Model Implementation for PPE Detection
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvLayer(nn.Module):
    """Convolutional layer with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for YOLOv3."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = ConvLayer(out_channels // 2, out_channels, kernel_size=3)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


class DarknetBlock(nn.Module):
    """Darknet block for YOLOv3 backbone."""
    
    def __init__(self, in_channels: int, out_channels: int, num_residuals: int):
        super(DarknetBlock, self).__init__()
        layers = [ConvLayer(in_channels, out_channels, kernel_size=3, stride=2)]
        
        for _ in range(num_residuals):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class YOLOv3(nn.Module):
    """
    YOLOv3 Model for PPE Detection.
    Detects PPE items (helmets, vests) in images.
    """
    
    def __init__(self, num_classes: int = 3, num_anchors: int = 3):
        """
        Initialize YOLOv3 model.
        
        Args:
            num_classes: Number of PPE classes
            num_anchors: Number of anchor boxes per scale
        """
        super(YOLOv3, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_size = num_anchors * (num_classes + 5)  # 5 = 4 bbox coords + 1 objectness
        
        # Backbone - Darknet-53
        self.layer1 = ConvLayer(3, 32, kernel_size=3, stride=1)
        self.layer2 = DarknetBlock(32, 64, 1)
        self.layer3 = DarknetBlock(64, 128, 2)
        self.layer4 = DarknetBlock(128, 256, 8)
        self.layer5 = DarknetBlock(256, 512, 8)
        self.layer6 = DarknetBlock(512, 1024, 4)
        
        # Detection heads
        self.detection_head_1 = self._build_detection_head(1024, self.output_size)
        self.detection_head_2 = self._build_detection_head(512, self.output_size)
        self.detection_head_3 = self._build_detection_head(256, self.output_size)
    
    def _build_detection_head(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build detection head."""
        return nn.Sequential(
            ConvLayer(in_channels, in_channels * 2, kernel_size=3),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through YOLOv3.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of detection outputs at different scales
        """
        # Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        scale3 = x
        x = self.layer4(x)
        
        scale2 = x
        x = self.layer5(x)
        
        scale1 = x
        x = self.layer6(x)
        
        # Detection heads
        out1 = self.detection_head_1(x)
        out2 = self.detection_head_2(scale1)
        out3 = self.detection_head_3(scale2)
        
        return out1, out2, out3


class YOLOv3Lite(nn.Module):
    """
    Lightweight YOLOv3 for faster inference.
    Suitable for mobile and edge devices.
    """
    
    def __init__(self, num_classes: int = 3):
        super(YOLOv3Lite, self).__init__()
        
        self.num_classes = num_classes
        
        # Lightweight backbone
        self.features = nn.Sequential(
            ConvLayer(3, 32, kernel_size=3, stride=2),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            ConvLayer(128, 256, kernel_size=3, stride=2),
            ConvLayer(256, 512, kernel_size=3, stride=2),
        )
        
        # Detection layers
        self.detection = nn.Sequential(
            ConvLayer(512, 256, kernel_size=3),
            nn.Conv2d(256, (num_classes + 5) * 3, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through lightweight YOLOv3.
        
        Args:
            x: Input tensor
            
        Returns:
            Detection output
        """
        x = self.features(x)
        x = self.detection(x)
        return x
