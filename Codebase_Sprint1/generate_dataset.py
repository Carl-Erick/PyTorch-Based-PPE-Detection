"""
Sample data generation script for PPE Detection dataset
"""

import os
import numpy as np
from PIL import Image
import random


def create_sample_dataset(output_dir: str = 'Dataset_Sample', num_samples: int = 20):
    """
    Create sample dataset with synthetic images and annotations.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of sample images to generate
    """
    
    # Create directory structure
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'annotations')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    ppe_classes = ['helmet', 'vest', 'no_ppe']
    
    print(f"Generating {num_samples} sample images...")
    
    for i in range(num_samples):
        # Generate random image
        height, width = 480, 640
        img_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        
        # Add some structure (simple rectangles to simulate objects)
        num_objects = random.randint(1, 3)
        annotations = []
        
        for _ in range(num_objects):
            class_id = random.randint(0, len(ppe_classes) - 1)
            
            # Random bounding box
            box_w = random.randint(50, 200)
            box_h = random.randint(50, 200)
            x_min = random.randint(0, width - box_w)
            y_min = random.randint(0, height - box_h)
            x_max = min(x_min + box_w, width)
            y_max = min(y_min + box_h, height)
            
            # Draw rectangle on image
            color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])
            img_array[y_min:y_max, x_min:x_max] = color
            
            # Convert to YOLO format (normalized)
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w_norm = (x_max - x_min) / width
            h_norm = (y_max - y_min) / height
            
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Save image
        img = Image.fromarray(img_array, 'RGB')
        image_path = os.path.join(images_dir, f'image_{i:04d}.jpg')
        img.save(image_path)
        
        # Save annotations
        annotation_path = os.path.join(annotations_dir, f'image_{i:04d}.txt')
        with open(annotation_path, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    print(f"\nDataset created successfully!")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Classes: {', '.join(ppe_classes)}")
    print(f"  Total samples: {num_samples}")


if __name__ == '__main__':
    create_sample_dataset()
