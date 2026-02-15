"""
Main training script for PPE Detection
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.loader import PPEDataLoader
from src.models.ppemodel import PPEDetectionModel, SimplePPEClassifier
from src.models.yolov3 import YOLOv3
from src.train import Trainer
from src.utils.helpers import set_seed, count_parameters, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPE Detection Model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='simple',
        choices=['simple', 'resnet', 'yolov3'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        config = {}
    
    # Create model
    print(f"\nBuilding {args.model} model...")
    if args.model == 'simple':
        model = SimplePPEClassifier(num_classes=3)
    elif args.model == 'resnet':
        model = PPEDetectionModel(num_classes=3, backbone='resnet50')
    elif args.model == 'yolov3':
        model = YOLOv3(num_classes=3)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Create data loaders (dummy paths for demonstration)
    print("\nCreating data loaders...")
    # train_loader = PPEDataLoader.get_data_loader(
    #     image_dir='Dataset_Sample/images',
    #     annotation_dir='Dataset_Sample/annotations',
    #     batch_size=args.batch_size,
    #     shuffle=True
    # )
    # val_loader = PPEDataLoader.get_data_loader(
    #     image_dir='Dataset_Sample/images',
    #     annotation_dir='Dataset_Sample/annotations',
    #     batch_size=args.batch_size,
    #     shuffle=False
    # )
    
    print("Data loaders created (dataset not included in Sprint 1 demo)")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=3,
        learning_rate=args.learning_rate
    )
    
    # Train model
    print("\nNote: Training skipped - dataset not included in Sprint 1")
    print("To train: Provide training and validation datasets")
    
    # This would be called once data loaders are available:
    # trainer.fit(train_loader, val_loader, num_epochs=args.epochs)
    
    print("\nTraining script initialized successfully!")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")


if __name__ == '__main__':
    main()
