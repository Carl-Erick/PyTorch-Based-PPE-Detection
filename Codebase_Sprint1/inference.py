#!/usr/bin/env python
"""
Inference script for PPE Detection
"""

import torch
import argparse
from pathlib import Path
import cv2
import numpy as np

from src.models.ppemodel import SimplePPEClassifier
from src.data.preprocessing import PreprocessingPipeline
from src.utils.helpers import get_device


class PPEDetector:
    """
    Inference class for PPE detection.
    """
    
    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize PPE detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
        """
        self.device = device or get_device()
        self.model = SimplePPEClassifier(num_classes=3)
        
        # Load checkpoint
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}, using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = PreprocessingPipeline()
        self.class_names = ['helmet', 'vest', 'no_ppe']
    
    def predict(self, image_path: str) -> dict:
        """
        Predict PPE class for an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        image_tensor = self.preprocessor.preprocess_image(image, return_tensor=True)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        result = {
            'image_path': image_path,
            'prediction': self.class_names[prediction],
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict PPE class for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="PPE Detection Inference")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory containing images'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    device = get_device()
    detector = PPEDetector(args.model, device)
    
    # Run inference
    if args.image:
        result = detector.predict(args.image)
        print("\nPrediction Result:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print("  Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
    
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        results = detector.predict_batch(image_paths)
        print(f"\n{'='*60}")
        print(f"Batch Inference Results ({len(results)} images)")
        print(f"{'='*60}")
        
        for result in results:
            print(f"\nImage: {result['image_path']}")
            if 'error' not in result:
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
    
    else:
        print("Please specify either --image or --image_dir")


if __name__ == '__main__':
    main()
