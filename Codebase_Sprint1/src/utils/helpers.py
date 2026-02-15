"""
Helper utility functions
"""

import torch
import random
import numpy as np
from typing import Dict, Optional
import os


def set_seed(seed: int = 42):
    """
    Set seed for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get the available device (GPU or CPU).
    
    Returns:
        torch.device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler = None,
    extra_dict: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler
        extra_dict: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_dict is not None:
        checkpoint.update(extra_dict)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device,
    scheduler = None
):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load checkpoint to
        scheduler: Learning rate scheduler
        
    Returns:
        Dictionary with checkpoint information (epoch, loss, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    
    return checkpoint


def freeze_backbone(model: torch.nn.Module, num_layers_to_freeze: int):
    """
    Freeze backbone layers for transfer learning.
    
    Args:
        model: Model with backbone
        num_layers_to_freeze: Number of layers to freeze
    """
    layers = list(model.backbone.children())
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: torch.nn.Module):
    """
    Unfreeze all backbone layers.
    
    Args:
        model: Model with backbone
    """
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0):
    """
    Clip gradients to prevent explosion.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
